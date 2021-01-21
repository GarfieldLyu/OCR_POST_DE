# OCR_POST_DE
OCR post correction for old German corpus 
Libraries:
  python 3.7
  keras 2.4.3, tensorflow 2.3.1, pytorch 1.4.0
  Other packages: NLTK, numpy, gensim, datasketch, Bio.pairwise2
 
create_data:
  1. Download OCRed book from ÖNB by the unique barcode, see dataScrapy.py.
  2. Clean the downloaded raw text, see parseText.py
  3. Providing a downloaded OCRed book and the corresponding transcription from DTA, see sentenceAlignment.py
  already generated sentence pairs (ocr_seq, trans_seq) and the original books are under PKL/
  
CRF:

keras_implement:

torch_implement:


  
