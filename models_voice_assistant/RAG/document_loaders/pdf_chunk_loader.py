from typing import List
from pypdf import PdfReader
from os import listdir
from os.path import isfile, join

class PDFChunkLoader():

  def __gen_split_overlap(self, text: str, size: int, overlap = 0):        
    if size < 1 or overlap < 0:
      raise ValueError('size must be >= 1 and overlap >= 0')
            
    for i in range(0, len(text) - overlap, size - overlap):            
      yield text[i:i + size]

  def get_pdf_files(self, path: str) -> List[str]:

    return [f for f in listdir(path) if isfile(join(path, f)) and f.endswith(".pdf")]

  def load_pdf_split(self, path: str, min_chunk_size: int = 200, split_char: str | None = None) -> List[str]:
    reader = PdfReader(path)
    data = []
    text = ""

    for page in reader.pages:
      text += page.extract_text()
      
    if split_char != None:
      to_store = ""
      for chunk in text.split(split_char):
        to_store += chunk
        if len(to_store) > min_chunk_size:
          data.append(to_store)
          to_store = ""
      data.append(to_store)
    else:
      for chunk in self.__gen_split_overlap(text, min_chunk_size, 50):
        data.append(chunk)

    return data