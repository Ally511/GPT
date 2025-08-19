"""BPE function

function to create byte pairs based off a dictionary of tokens and their counts.
"""
import numpy as np
from collections import defaultdict
from collections import OrderedDict
from collections import Counter

from utility_functions import performance,find_top_indices,generate_n_grams
from generator import to_byte_pair

def bpe(dictionary, k):
  """
  Input:
    dictionary (dict): a dictionary that contains the tokens and their respective counts
  return:
    vocab_bpe (list): vocabulary of the corpus
    sorted_token_freq:
    dict_matrix:
  """
  # get all unique characters in original corpus
  all_keys = "_ ".join(dictionary.keys())
  vocab_bpe = list(set(all_keys))

  # Corpus/dictionary in einzelne tokens splitten, nach jedem Wort (VOR space!) "_" einfügen
    # worte als list of characters
  dict_matrix = []
  for key, value in dictionary.items():
    new_key = list(f"{str(key)}_ ")
    dict_matrix.append([new_key, value])

  # NICHT corpus, sondern liste an Wörtern in einzelne tokens splitten,
  # >> jede occurence mit counts der Worte multiplizieren
  #token_freq = defaultdict(int) # TODO: does that need to be moved outside of the loop?
  iteration = True
  num_rounds = 0
  while iteration:
    token_freq = defaultdict(int) # ACHTUNG: moved this inside again
    for token_list, value in dict_matrix:
      for i in range(len(token_list)-2):
        # wollen den und den nächsten token als key
        search_key = token_list[i] + token_list[i+1]
        # zu dictionary hinzufügen falls key noch nicht existiert
        token_freq[search_key] += value

    # word_freqs: gehen jede existierende Folge aus zwei tokens in list of words von vorne bis hinten durch
    c = Counter(token_freq)
    sorted_token_freq = {key: value for key, value in sorted(
        c.items(), key=lambda item: item[1], reverse=True)}

    # Find the most frequent token not already in vocab_bpe
    for token in sorted_token_freq.keys():
      if token not in vocab_bpe:
        first_token = token
        break
      else:
        first_token = None  # Optional: fallback in case all tokens are already in vocab

    if first_token:
      vocab_bpe.append(first_token)
    else:
      print("No new token to add.")

    for i in range(len(dict_matrix)):
        token_list, value = dict_matrix[i]
        j = 0
        while j < len(token_list) - 1:
            search_key = token_list[j] + token_list[j + 1]
            if search_key == first_token:
                merged_token = search_key
                # Merge the tokens
                token_list = token_list[:j] + [merged_token] + token_list[j + 2:]
                # Don't increment j — might be able to merge again
            else:
                j += 1
        dict_matrix[i][0] = token_list
    if k:
       k -= 1
       iteration = (k > 0)

    else:
       num_rounds += 1
       accuracy = performance(dictionary, vocab_bpe, 500)
       if accuracy > 70: # ACHTUNG: changed this line
          iteration = False
       #iteration != (accuracy > 70)

       if num_rounds > 1500:
          print("exceeded, accuracy: ", accuracy)
          iteration = False

  return vocab_bpe, sorted_token_freq, dict_matrix


def get_best_merges(dict_train, text_train, dict_valid,text_valid,min_k,max_k,step):

  ks = []
  n_grams = []
  perplexities = []

  for k in range (min_k,max_k,step):
    vocab_train, sorted_token_freq_train, dict_matrix_train = bpe(dict_train,k)
    n_gram_train = to_byte_pair(text_train, vocab_train)
    vocab_valid, sorted_token_freq_valid, dict_matrix_valid = bpe(dict_valid,k)
    # changed this to vocab_valid since we want to see how well the ngram performs on the train vocab
    n_gram_valid = to_byte_pair(text_valid, vocab_train)

    our_n_grams_valid = generate_n_grams(n_gram_train,4, len(vocab_train))
    n_gram_num = 0

    for n_gram in our_n_grams_valid:
      n_gram_num += 1
      print(f"N-gram split for {n_gram_num}-gram, k = {k}: {n_gram.split_text[:10]}")
      n_gram.old_perplexity(n_gram_valid)
      perplexity = n_gram.perplexity(n_gram_valid)
      perplexities.append(perplexity)
      ks.append(k)
      n_grams.append(n_gram_num)

  top_indices = find_top_indices(perplexities, 3)
  
  """out   = []
  for ix in top_indices:
    out.extend((ks[ix], perplexities[ix], ngrams[ix]))
  return tuple(out)  # (k1,p1,n1, k2,p2,n2, k3,p3,n3)"""
  
  best_k = ks[top_indices[0]]
  best_perplexity = perplexities[top_indices[0]]
  best_n_gram = n_grams[top_indices[0]]

  second_best_k = ks[top_indices[1]]
  second_best_perplexity = perplexities[top_indices[1]]    
  second_best_n_gram = n_grams[top_indices[1]]

  third_best_k = ks[top_indices[2]]
  third_best_perplexity = perplexities[top_indices[2]]
  third_best_n_gram = n_grams[top_indices[2]]

  return best_k,best_perplexity,best_n_gram,second_best_k,second_best_perplexity,second_best_n_gram,third_best_k,third_best_perplexity,third_best_n_gram,ks,n_grams,perplexities