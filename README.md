# OCR_POST_DE

OCR post correction for old German corpus. More details can be found in our paper(https://arxiv.org/abs/2102.00583). 

Libraries:
  python 3.7
  keras 2.4.3, tensorflow 2.3.1, pytorch 1.4.0
  
  Other packages: NLTK, numpy, gensim, datasketch, Bio.pairwise2, entmax
 
create_data:
  1. Download OCRed book from ÖNB(https://iiif.onb.ac.at/gui/manifest.html) by the unique barcode, see dataScrapy.py.
  2. Clean the downloaded raw text, see parseText.py
  3. Providing a downloaded OCRed book and the corresponding transcription from DTA, see sentenceAlignment.py
  already generated sentence pairs (ocr_seq, trans_seq) and the original books are under PKL/
  
  
CRF (conditional random field):

  There are many cases that the OCR quality is acceptible(e.g, books from 18_th and later centuries), most of the errors are from segmentation instead of character misrecognition. We provide a tagger trained from the German wikipedia corpus, use CRF to correct segmentation errors only.
  See word_segment.py for details. Due to the space limitation, you can utilize the source code and your own data to train the tagger, alternately you can also download the trained tagger(dewiki_segmentation.crfsuite) from https://drive.google.com/file/d/1h7mwsXERKrymGnVNfYuDcOGf25DbohEp/view?usp=sharing


keras_implement:

  The original implementation is based on keras, see networks.py for all models definition. The ocr_corrector.py contains all functions to train, evaluate, generate output for a single sentence or in batch level.


torch_implement:

  A torch implementation is also on going, for now we provide a standard attention based encoder-decoder model. The only differences are:
    1. a random teacher-forcing training.
    2. entmax (instead of softmax).
  These two changes boosted the performance further, which also maintains the simplicity. 
  See Model.py for model definition, see seq2seq.py for training, evaluation, generating and other utilities. A trained model with all instances and the data   pairs can be found from https://drive.google.com/drive/folders/1qBI-2IhYPBGtMcGVWGb19lJCu0jV7QId?usp=sharing


If more questions related to the code or data, please contact us. 
Please cite our paper if you find it useful. 
```
@misc{lyu2021neural,
      title={Neural OCR Post-Hoc Correction of Historical Corpora}, 
      author={Lijun Lyu and Maria Koutraki and Martin Krickl and Besnik Fetahu},
      year={2021},
      eprint={2102.00583},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
