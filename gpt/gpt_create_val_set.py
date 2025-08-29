import numpy as np
import json
with open("../corpora/vocab_2nd.json", "r", encoding="utf-8") as f:
    vocab_train = json.load(f)

with open("../corpora/Shakespeare_2nd_best_merge_valid.txt", "r") as f:
    shakespeare_byte_valid = f.read().split() 

indx_path = "indices_2nd_text_valid"
data_path = "Shakespeare_2nd_merge_valid"


vocab = vocab_train
indices = np.arange(0,len(vocab),1)
inidces = indices.astype(int)
indices = indices.tolist()
key_byte = dict(zip(vocab, indices))
value_byte = dict(zip(indices,vocab))

# Map each token in shakespeare_byte_train to its index using key_byte
indices_translation = [key_byte[token] for token in shakespeare_byte_valid if token in key_byte]

with open(f'../corpora/{indx_path}.txt', 'w') as indices_text_val:
    indices_text_val.write(str(indices_translation))

with open (f"../corpora/{indx_path}.txt", 'r') as f:
  indices_text_val = eval(f.read())


bytes_translation = [token for token in indices_text_val if token in value_byte]
print(bytes_translation[:10])
with open(f'../corpora/{data_path}.txt', 'w') as bytes_text_val:
    bytes_text_val.write(str(bytes_translation))