import matplotlib.pyplot as plt

def plot_tokenLength_dist(tokens, title):
    lengths = [len(token) for token in tokens]
    plt.figure(figsize=(10, 5))
    plt.hist(lengths, bins=20, edgecolor='black')
    plt.title(title)
    plt.xlabel("Token Length")
    plt.ylabel("Frequency")
    plt.show()