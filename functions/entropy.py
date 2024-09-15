import numpy as np
from collections import Counter

# Shannon Entropy
def calculate_entropy(frequencies):
    total = sum(frequencies.values())
    probs = np.array(list(frequencies.values())) / total
    # normalizing
    probs = probs / np.sum(probs)
    log_prob = np.log(probs) / np.log(2)
    # Calculate the entropy
    entropy = -np.sum(probs * log_prob)
    return entropy

# Get top N tokens by entropy
def get_top_tokens(tokens):
    frequencies = Counter(tokens)
    # Calculating entropy for the entire distribution
    tokens_entropy = calculate_entropy(frequencies)

    # Sort tokens by frequency and select the top 1000
    top_tokens = [token for token, _ in frequencies.most_common(1000)]

    return top_tokens, tokens_entropy