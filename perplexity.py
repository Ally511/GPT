import generator.to_byte_pair

# load the test text 
with open('test.txt', 'r') as file:
    test_text = file.read()


def perplexity(text, ngram, dict_tokens, vocab):
    """
    Calculate perplexity of a given text based on an n-gram model.
    
    Args:
        text (str): Input text string.
        ngram (ndarray): N-dimensional array of n-gram probabilities.
        dict_tokens (dict): Mapping from token -> token index.
        vocab (list): Vocabulary list used for tokenization.
        generator (object): Object containing 'to_byte_pair' method for tokenization.

    Returns:
        float: Perplexity value of the input text.
    """
    text = generator.to_byte_pair.to_byte_pair(text,vocab)
    n = ngram.ndim  # order of the n-gram model
    
    prob_total = 1.0
    # Iterate through the text, considering n-grams
    for t in range(len(text) - n + 1):
        tokens = text[t:t + n]
        if any(token not in dict_tokens for token in tokens):
            # Assign very small probability for unknown n-grams (smoothing)
            prob = 1e-10
        else:
            # Map tokens to indices
            indices = [dict_tokens.get(token) for token in tokens]
            # retrieve the probability of the n-gram
            prob = ngram[tuple(indices)]
            prob_total *= prob

    #calculate perplexity
    perplexity_value = prob_total ** (-1 / len(text))
    
    return perplexity_value
        