import numpy as np
import operator
import random

def to_byte_pair(context, vocab):
    """
    Tokenize a string into subword units using a simple greedy byte-pair-like algorithm.

    Args:
        context (str): The input string to tokenize.
        vocab (list of str): Known subword tokens, e.g. from a BPE vocabulary.

    Returns:
        list of str: The tokenized input as a list of matched tokens and any leftover characters.
    """
    # Replace spaces with underscores for consistent token matching
    context = context.lower()
    context += "_"
    context = context.replace(' ', '_')
    punctuation = [ '.', '!', '?', ':', ';']
    def remove_punctuation(input_string):
        result = input_string
        for char in punctuation:
            if char != ' ':
                result = result.replace(char, f'_{char}')
        return result

    context = remove_punctuation(context)


    # Sort vocabulary by length (longer matches first)
    vocab.sort(key=lambda x: len(x), reverse=True)

    final_list = []
    mismatch = ""

    while context != "":
        old_context = context
        match = False

        # Try to find the longest matching token at the current position
        for token in vocab:
            if token == context[:len(token)]:
                if mismatch:
                    final_list.append(mismatch)  # Add any mismatched leftover
                final_list.append(token)
                context = context[len(token):]  # Remove matched part
                mismatch = ""
                match = True
                break

        # If no match, accumulate unmatched character(s)
        if not match:
            mismatch += context[0]
            context = context[1:]

        # Safety: break to prevent infinite loop
        if old_context == context:
            break

    # Add any leftover mismatches at the end
    if mismatch:
        final_list.append(mismatch)

    return final_list


def generate_new(context, ngram_dicts, n, vocab, max_len=200, seed=0):
    """
    ngram_dicts: list where
      ngram_dicts[0] = unigram {token -> prob}
      ngram_dicts[1] = bigram  {(t1,) -> {t2 -> prob}}
      ngram_dicts[2] = trigram {(t1,t2) -> {t3 -> prob}}
      ...
    n: order to generate with (1..len(ngram_dicts))
    """
    rng = random.Random(seed)

    def sample_from_dist(dist):
        # dist is {token: prob}; renormalize to sum 1 (your Laplace leaves mass for unseen tokens)
        toks, probs = zip(*dist.items())
        probs = np.array(probs, dtype=float)
        probs = probs / probs.sum()
        # numpy choice needs a NumPy RNG; convert index picked by cumulative sampling:
        c = probs.cumsum()
        r = rng.random()
        i = int((c > r).argmax())
        return toks[i]

    # tokenize starting context
    text = to_byte_pair(context, vocab)

    end_tokens = ['.', ':', '?', '!', '._', '!_', '?_', ':_', ';_']

    def backoff(m, keys):
        """Try order m, else back off to m-1 with a shortened key."""
        if m == 1:
            # unigram sample
            return sample_from_dist(ngram_dicts[0])

        # context for this order = last (m-1) tokens
        ctx = tuple(keys[-(m-1):]) if (m-1) > 0 else tuple()
        dist = ngram_dicts[m-1].get(ctx)

        if dist:  # found this context
            return sample_from_dist(dist)
        else:
            # back off to lower order, keeping the same *full* history,
            # but backoff() will reslice it to the right length
            return backoff(m-1, keys)

    next_token = ''
    while next_token not in end_tokens and len(text) <= max_len:
        next_token = backoff(n, text)
        text.append(next_token)

    pretty_text = "".join(str(x) for x in text).replace('_', ' ')
    return pretty_text




def generate(context, ngrams, n, vocab):
    """
    Generate text using an n-gram given a defined context.

    Args:
        context (str): The initial context string.
        ngram (np.ndarray): The n-dimensional n-gram tensor. Shape = (V, V, ..., V)
        # dict_tokens (dict): Mapping from token -> index.
        # dict_in (dict): Mapping from index -> token.
        vocab (list of str): Subword tokens used by the tokenizer.

    Returns:
        str: The generated text with underscores replaced by spaces.
    """
    # n = ngram.ndim  # order of the n-gram model
    ngram = ngrams[n-1] # ensure it's a NumPy array (redundant if it already is)

    # define end_tokens as punctuation
    end_tokens = ['.', ':', '?', '!','._', '!_', '?_', ':_', ';_']

    # Tokenize input context
    text = to_byte_pair(context, vocab)

    next_word = ''
    while next_word not in end_tokens and len(text) <= 200:
        # Get the last (n-1) or n tokens for context (depending on your convention)
        # print(text)
        context = text[-(n-1):]

        preceding_keys = tuple(context)

        def backoff(n, ngrams, keys):
            #print(keys)
            if keys in ngrams[n-1].keys():
                
                sorted_dict = {key: value for key, value in sorted(ngrams[n-1][keys].items(), key=lambda item: item[1], reverse=True)}
                # print(sorted_dict)
                zufall = random.randint(0,(len(sorted_dict)-1)//2)
                next_token = list(sorted_dict.keys())[zufall]
                return next_token
            
            else:
                #print("wrong!!!")
                # CHANGED
                """if n == 1:
                    return 'my_'"""
                if n == 1:
                    sorted_dict = {key: value for key, value in sorted(ngrams[0].items(), key=lambda item: item[1], reverse=True)}
                    zufall = random.randint(0,(len(sorted_dict)-1)//2)
                    return list(sorted_dict.keys())[zufall]
                return backoff(n-1, ngrams, keys)

        next_word = backoff(n, ngrams, preceding_keys)
        text.append(next_word)
        
    pretty_text = "".join(str(x) for x in text)
    pretty_text = pretty_text.replace('_', ' ')
    return pretty_text