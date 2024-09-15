import time
from functions.parallel_tokenize import parallel_tokenize
from functions.tokenizers import tokenize_huggingface

def tokenizeBert_ST(cleaned_text):
    # Perform tokenization in parallel way
    time_st = time.time()
    huggingface_tokens_STmerged = parallel_tokenize(cleaned_text, tokenize_huggingface, num_workers=61)
    time_ed = time.time()
    time_bert_STmerged = time_ed - time_st
    return huggingface_tokens_STmerged, time_bert_STmerged

def tokenizeBert(cleaned_text):
    time_st = time.time()
    huggingface_tokens = parallel_tokenize(cleaned_text, tokenize_huggingface, num_workers=61, merge=False)
    time_ed = time.time()
    time_bert = time_ed - time_st
    return huggingface_tokens, time_bert