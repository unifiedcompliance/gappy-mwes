import spacy
import textacy

def mwe(text: str, start: int, start_char: int, end_char: int, pos: str, upos: str, mwe_tag:str):
  chunk = {
        "index": start + 1,
        "word": text,
        "pos": pos,
        "upos": upos,
        "mwe": mwe_tag,
        "characterOffsetBegin": start_char,
        "characterOffsetEnd": end_char
  }
  return chunk


def nounChunks(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    #mwes = []
    noun_chunk_mwe = {}
    noun_chunk_indices = {}
    phrases = textacy.extract.noun_chunks(doc, drop_determiners=True)
    noun_chunks_list = []
    for n in phrases:  
      span = doc[n.start:n.end]
      if len(span) > 1:
          noun_chunks_list.append(n.text)
          t = []
          indices = []
          for i, token in enumerate(span):
                if i == 0:
                    mwe_tag = 'B'
                else:
                    mwe_tag = 'I'
                t.append(mwe(token.text, token.i, token.idx, (token.idx + len(token.text)), token.pos_, token.tag_, mwe_tag))
                indices.append(token.i+1)
          noun_chunk_mwe[n.text] = t
          noun_chunk_indices[n.text] = indices
    
    return noun_chunks_list, noun_chunk_mwe, noun_chunk_indices
