import contractions
import re
import string
from concurrent.futures import ProcessPoolExecutor
from functions.tokenizers import spacy_tokenizer
stop_words = spacy_tokenizer.Defaults.stop_words

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = contractions.fix(text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()
    tokens = [token for token in tokens if token not in stop_words]
    return " ".join(tokens)

# Parallel text cleaning and tokenization.
# Utilize multi threaded process to clean the texts on multiple threads or "processes" for faster and parallel computation
def clean_parallelly(texts, num_workers=61):
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        cleaned_sentences = list(executor.map(clean_text, texts))
    return cleaned_sentences