"""
# 1. Load the tokenizers
"""
from transformers import BertTokenizerFast
import spacy

huggingface_tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased',clean_up_tokenization_spaces=True)
spacy_tokenizer = spacy.blank("en")

# Tokenization functions
def tokenize_huggingface(text):
    return huggingface_tokenizer.tokenize(text)

def tokenize_spacy(text):
    return [token.text.lower() for token in spacy_tokenizer(text)]