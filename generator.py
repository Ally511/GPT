import numpy as np

def to_byte_pair(context, vocab):
    # to do:
    # catch evil combinations
    # abbruchbedingung
    context = context.replace(' ', '_')
    # sort vocab
    vocab.sort(key=lambda x: len(x), reverse=True)
    final_list = []
    mismatch = ""
    while context != "":
        old_context = context
        match = False
        for token in vocab:
            if token == context[:len(token)]:
                final_list.append(mismatch)
                final_list.append(token)
                context = context[len(token):]
                mismatch = ""
                match = True
                break
        if not match:
            mismatch += context[0]
            context = context[1:]

        if old_context == context:
            break

    final_list.append(mismatch)
    return final_list


def generate(context, ngram, dict_tokens, dict_in, vocab):
    n = ngram.ndim
    ngram = np.array(ngram)
    end_tokens = ['.', ':', '?', '!']
    # tokenize
    text = to_byte_pair(context)

    next_word = ''
    while next_word not in end_tokens:
        context = text[-n:]

        indices = [None if token not in dict_tokens.keys() else dict_tokens[token] for token in context]

        if None in indices:
            next_idx = ngram.argmax()
        else:
            # build index to slice the tensor
            index = tuple(indices) + (slice(None),)

            # select the column
            # potentially this requires ngram to be a tensor
            column = ngram[index]

            # get the index of the most likely next token
            next_idx = column.argmax()

        next_word = dict_in[next_idx]
        text.append(next_word)

    pretty_text = str(text)
    pretty_text = pretty_text.replace('_', ' ')
    return pretty_text

vocab = ["hallo_", "ha", "llo_", "Welt!"]
context = "hallo dcWelt! xsl"

to_byte_pair(context, vocab)
