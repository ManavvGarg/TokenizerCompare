import time
from functions.parallel_tokenize import parallel_tokenize
from functions.tokenizers import tokenize_spacy

def tokenizeSpacy(cleaned_text):
    time_st = time.time()
    spacy_tokens = parallel_tokenize(cleaned_text, tokenize_spacy, num_workers=61)
    time_ed = time.time()
    time_spacy = time_ed - time_st
    return spacy_tokens, time_spacy