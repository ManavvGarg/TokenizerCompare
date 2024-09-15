def comparer(comparison_df):
    # Initialize scores dictionary
    scores = {col: 0 for col in comparison_df.columns if col != 'Metric'}

    # Iterate over each metric row
    for index, row in comparison_df.iterrows():
        metric = row['Metric']
        values = row[1:].tolist()  # Get values for BERT and spaCy
        max_value = max(values)

        # Add +1 to the score for the tokenizer with the best performance in this metric
        for i, tokenizer in enumerate(scores.keys()):
            if values[i] == max_value:
                scores[tokenizer] += 1

    # Find the best tokenizer based on the highest score
    return scores