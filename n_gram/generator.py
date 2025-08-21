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
                print("wrong!!!")
                if n == 1:
                    return 'my_'
                return backoff(n-1, ngrams, keys)


        next_word = backoff(n, ngrams, preceding_keys)
        # Map tokens to indices; if unknown, use None
        # indices = [dict_tokens.get(token) for token in context]
        #
        # if None in indices:
        #     # If unknown tokens: fallback to overall next-token distribution
        #     column = ngram.sum(axis=tuple(range(n - 1)))
        #     next_idx = column.argmax()
        # else:
        #     # Build index tuple: fix context, free last axis
        #     index = tuple(indices) + (slice(None),)
        #
        #     # Get conditional next-token distribution
        #     column = ngram[index]
        #
        #     next_idx = column.argmax()

        # next_word = dict_in[next_idx]
        # text.append(next_word)
        
        text.append(next_word)
        # Join final tokens and clean up
        # pretty_text = ''.join(text)
        # pretty_text = pretty_text.replace('_', ' ')
        #print(pretty_text)
        # print(text)
    # text = str(text)
    # Join final tokens and clean up
    pretty_text = "".join(str(x) for x in text)
    # pretty_text = ''.join(text)
    pretty_text = pretty_text.replace('_', ' ')
    return pretty_text