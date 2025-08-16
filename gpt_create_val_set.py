import numpy as np

with open (r"corpora/Shakespeare_byte_valid.txt", 'r') as f:
   shakespeare_byte_valid = eval(f.read())

with open (r"corpora/vocab_train.txt", 'r') as f:
   vocab_train = eval(f.read())

vocab = vocab_train
indices = np.arange(0,len(vocab),1)
inidces = indices.astype(int)
indices = indices.tolist()
key_byte = dict(zip(vocab, indices))
value_byte = dict(zip(indices,vocab))

# Map each token in shakespeare_byte_train to its index using key_byte
indices_translation = [key_byte[token] for token in shakespeare_byte_valid if token in key_byte]

with open('corpora/indices_text_valid.txt', 'w') as indices_text_val:
    indices_text_val.write(str(indices_translation))

with open (r"corpora/indices_text_valid.txt", 'r') as f:
  indices_text_val = eval(f.read())


bytes_translation = [value_byte[token] for token in indices_text_val if token in value_byte]

with open('corpora/bytes_text_valid.txt', 'w') as bytes_text_val:
    bytes_text_val.write(str(bytes_translation))