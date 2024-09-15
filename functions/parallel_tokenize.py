from concurrent.futures import ProcessPoolExecutor
from functions.mergeST import merge_subtokens

# Utilize multi threaded process to tokenize the texts on multiple threads or "processes" for faster and parallel computation
def parallel_tokenize(texts, tokenize_function, num_workers=128, merge=True):
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        tokens = list(executor.map(tokenize_function, texts))
    if merge == True:
        return merge_subtokens([token for sublist in tokens for token in sublist])
    else:
        return [token for sublist in tokens for token in sublist]