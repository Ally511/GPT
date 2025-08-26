import numpy as np

def get_batch(input, batch_size, chunk_size):
    """ Creates random batches of sequential text chunks and their shifted targets.
    @:param
    input (list or np.ndarray): tokenized text data as a sequence of integers
    batch_size (int): size of each batch
    chunk_size (int): length of each sequence that is contained in the batch

    @:returns
    input_batch (np.ndarray): input sequences of shape (batch_size, chunk_size)
    target_batch (np.ndarray): target sequences are the input sequences shifted by one position with shape (batch_size, chunk_size)

    """
    input_batch = []
    target_batch = []
    idx = np.random.randint(0, len(input) - (chunk_size + 1), size=batch_size)
    for i in range(0, len(idx)):
        input_batch.append(input[idx[i]:idx[i] + chunk_size])
        target_batch.append(input[idx[i] + 1:idx[i] + (chunk_size + 1)])

    input_batch = np.array(input_batch)
    target_batch = np.array(target_batch)

    return input_batch, target_batch


def decode_characters(input, vocab_train):
    """Decodes a list of indices back to their corresponding characters
    given the above defined vocabulary

    @:param
    input (list or np.ndarray): a list of token IDs to decode
    vocab_train (list): the vocabulary list containing the tokens

    @:returns
    decoded (str): the decoded string
    """
    vocab = vocab_train
    indices = np.arange(0, len(vocab), 1)
    inidces = indices.astype(int)
    indices = indices.tolist()
    key_byte = dict(zip(vocab, indices))
    value_byte = dict(zip(indices, vocab))

    decoded = [] #given the input, we will decode it back to characters
    for i in range(0,len(input)):
        decoded.append(value_byte[input[i]])#using the translation dctionary: value_byte
    #make its prettier by joining list to actual words and replacing underscores with spaces
    decoded = ''.join(decoded)
    decoded = decoded.replace('_', ' ')
    return decoded
