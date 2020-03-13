import os
import random
import numpy as np
import torch
from seqeval.metrics import f1_score, precision_score, recall_score
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler

pad_token_label_id = CrossEntropyLoss().ignore_index
MAX_LENGTH=512
labels = ['B', 'I', 'O']

class Input(object):
    def __init__(self, guid, words):
        self.guid = guid
        self.words = words


def read_from_tokens(tokens):
    guid_index = 1
    examples = []

    for i in tokens:
        exm = Input(guid_index, i)
        examples.append(exm)
        guid_index += 1

    return examples



class featureinput(object):
    def __init__(self, input_ids, input_mask, segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids

def convert_input_to_features(examples, max_seq_length, tokenizer, pad_token=0,
                              cls_token_at_end=False, cls_token="[CLS]",
                              cls_token_segment_id=1, sep_token="[SEP]",
                              sep_token_extra=False, pad_on_left=False,
                              pad_token_segment_id=0, pad_token_label_id=-100,
                              sequence_a_segment_id=0, mask_padding_with_zero=True):

    label_map = {label: i for i, label in enumerate(labels)}
    features = []
    for (ex_index, example) in enumerate(examples):

        tokens = []
        label_ids = []
        for word in example.words:
            word_tokens = tokenizer.tokenize(word)
            tokens.extend(word_tokens)
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            label_ids.extend([0] + [pad_token_label_id] * (len(word_tokens) - 1))

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = 3 if sep_token_extra else 2
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            label_ids = label_ids[: (max_seq_length - special_tokens_count)]

        tokens += [sep_token]
        label_ids += [pad_token_label_id]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
            label_ids += [pad_token_label_id]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens += [cls_token]
            label_ids += [pad_token_label_id]
            segment_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            label_ids = [pad_token_label_id] + label_ids
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            label_ids = ([pad_token_label_id] * padding_length) + label_ids
        else:
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length
            label_ids += [pad_token_label_id] * padding_length


        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length

        features.append(
            featureinput(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids, label_ids=label_ids)
        )
    return features

def load_input(tokens, tokenizer):
    # Load data features from cache or dataset file
    #print("Creating features from dataset file at %s", data_dir)

    input_token = read_from_tokens(tokens)
    features = convert_input_to_features(input_token, MAX_LENGTH, tokenizer, pad_token_label_id=pad_token_label_id)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return dataset

def predict(tokens, model, tokenizer):
    ds = load_input(tokens, tokenizer)

    n_gpu = 1
    per_gpu_eval_batch_size = 8
    eval_batch_size = per_gpu_eval_batch_size * max(1, n_gpu)

    eval_sampler = SequentialSampler(ds)
    eval_dataloader = DataLoader(ds, sampler=eval_sampler, batch_size=eval_batch_size)
    preds = None
    out_label_ids = None
    model.eval()

    for i,batch in enumerate(eval_dataloader):

        #batch = tuple(t.to(device) for t in batch)
        batch = tuple(t for t in batch)

        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            inputs["token_type_ids"] = batch[2]
            outputs = model(**inputs)
            logits = outputs[1]
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)


    preds = np.argmax(preds, axis=2)


    label_map = {i: label for i, label in enumerate(labels)}
    out_label_list = [[] for _ in range(out_label_ids.shape[0])]
    preds_list = [[] for _ in range(out_label_ids.shape[0])]

    for i in range(out_label_ids.shape[0]):
        for j in range(out_label_ids.shape[1]):
            if out_label_ids[i, j] != pad_token_label_id:
                out_label_list[i].append(label_map[out_label_ids[i, j]])
                preds_list[i].append(label_map[preds[i, j]])

    predictions = []
    for i,j in enumerate(tokens):
        for k in list(zip(j, preds_list[i])):
            predictions.append(k)
            print(k)

    return predictions
