# Function to merge sub-tokens
def merge_subtokens(tokens):
    merged_tokens = []
    current_word = ""

    for token in tokens:
        if token.startswith("##"):
            current_word += token[2:]
        else:
            if current_word:
                merged_tokens.append(current_word)
            current_word = token

    # Append the last word
    if current_word:
        merged_tokens.append(current_word)

    return merged_tokens