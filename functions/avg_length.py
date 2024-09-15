import numpy as np

def avg_length(tokens):
    return np.mean([len(token) for token in tokens])