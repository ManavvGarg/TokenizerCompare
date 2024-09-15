#import required lib
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse, JSONResponse
import pandas as pd
import warnings
warnings.filterwarnings("ignore", message=r"^None of")
warnings.filterwarnings("ignore", message=r"^Token indices")


"""
# 1. Loading the IMDB dataset
"""
from functions.dataset_loader import data_loader
train_dataset = data_loader()


"""
# 2. Text Preprocessing
    1. Lowering case of sentences
    2. Removal of URLs
    3. Removal of Contractions
    4. Removal of Punctuations
    5. Removal of Numbers
    6. Removal of Extra Spaces
    7. Removal of stop words
"""

from functions.clean_text import clean_parallelly

# Clean the train dataset in parallel
cleaned_text = clean_parallelly(train_dataset, num_workers=61)
        
"""
# 3. Tokenization
"""    
from functions.tokenize.bert import tokenizeBert, tokenizeBert_ST
from functions.tokenize.spacy import tokenizeSpacy

huggingface_tokens, time_bert = tokenizeBert(cleaned_text)
huggingface_tokens_STmerged, time_bert_STmerged = tokenizeBert_ST(cleaned_text)
spacy_tokens, time_spacy = tokenizeSpacy(cleaned_text)

"""
# 4. Calculating Entropy using Shannon's Entropy formula
"""

from functions.entropy import get_top_tokens

# Get the top tokens for all tokenizers
huggingface_top_tokens, huggingface_token_entropy = get_top_tokens([token for token in huggingface_tokens])
huggingface_top_tokens_STmerged, huggingface_token_entropy_STmerged = get_top_tokens([token for token in huggingface_tokens_STmerged])
spacy_top_tokens, spacy_token_entropy = get_top_tokens([token for token in spacy_tokens])

"""
# 5. Calculate overlap
"""

overlap = set(huggingface_top_tokens) & set(spacy_top_tokens)
overlap_percentage = len(overlap) / 1000 * 100

overlap_STmerged = set(huggingface_top_tokens_STmerged) & set(spacy_top_tokens)
overlap_percentage_STmerged = len(overlap) / 1000 * 100


"""
# 6. Calculate average token length
"""

from functions.avg_length import avg_length

huggingface_avg_length = avg_length(huggingface_top_tokens)
huggingface_avg_length_STmerged = avg_length(huggingface_top_tokens_STmerged)
spacy_avg_length = avg_length(spacy_top_tokens)

"""
# 7. Compare unique tokens
"""
# Sample of unique huggingface tokenizer(BERT) w/ subtokens (in comparison with spaCy)
huggingface_unique = set(huggingface_top_tokens) - set(spacy_top_tokens)

# Sample of unique huggingface tokenizer(BERT) w/o subtokens (in comparison with spaCy)
huggingface_unique_STmerged = set(huggingface_top_tokens_STmerged) - set(spacy_top_tokens)

# Sample of unique spaCy tokens (in comparison with BERT w/ subtokens)
spacy_unique = set(spacy_top_tokens) - set(huggingface_top_tokens)

#Sample of unique spaCy tokens (in comparison with BERT w/o subtokens)
spacy_unique_BERTst = set(spacy_top_tokens) - set(huggingface_top_tokens_STmerged)

"""
# 8. Calculate subword token percentage (for huggingface tokenizer(BERT) w/ subtokens)
"""
huggingface_subword_percentage = len([t for t in huggingface_top_tokens if t.startswith("##")]) / 1000 * 100

"""
# 9. Create a comparison DataFrame
"""
from functions.comparer import comparer

comparison_df = pd.DataFrame({
    'Metric': ['Overlap', 'Entropy','Avg Token Length', 'Subword Tokens', 'Tokenization Time','Unique Tokens'],
    'BERT w/ Subtokens': [overlap_percentage, huggingface_token_entropy, huggingface_avg_length, huggingface_subword_percentage, time_bert, len(huggingface_unique)],
    'BERT w/o Subtokens': [overlap_percentage_STmerged, huggingface_token_entropy_STmerged, huggingface_avg_length_STmerged, 0, time_bert_STmerged, len(huggingface_unique_STmerged)],
    'spaCy': [overlap_percentage, spacy_token_entropy, spacy_avg_length, 0, time_spacy, 0]
})

comparison_df_list = comparison_df.to_dict(orient="list")

# Calculate the weighted scores
scores = comparer(comparison_df)

# Find the best tokenizer based on weighted scores
best_tokenizer = max(scores, key=scores.get)

# app init
app = FastAPI()

@app.get("/", response_class=PlainTextResponse)
async def hello():
    try:
        return "Hello World! By manav garg"
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/tokenize", response_class=JSONResponse)
async def tokenize():
    result = {
        "Metric Data": comparison_df_list,
        "Comparison Result" : best_tokenizer
    }
    return JSONResponse(result)