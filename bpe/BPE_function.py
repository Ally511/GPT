"""BPE function

function to create byte pairs based off a dictionary of tokens and their counts.
"""
import numpy as np
from collections import defaultdict, Counter

from bpe.utility_functions import performance,find_top_indices,generate_n_grams, to_byte_pair

def bpe(dictionary, k):
  """
  Searches most common occurence of token following another. Merges them, and appends token to voabulary. 

  Input:
    dictionary (dict): a dictionary that contains the tokens and their respective occurence counts
    k (int): number of merges before returning
  return:
    vocab_bpe (list): vocabulary of the corpus
    sorted_token_freq: contains token frequencies after k merges, sorted
    dict_matrix: contains dictionary after k merges
  """
  # get all unique characters in original corpus
  all_keys = "_ ".join(dictionary.keys())
  vocab_bpe = list(set(all_keys))

  # split into single tokens: add "_" after each word, words as list of characters
  dict_matrix = []
  for key, value in dictionary.items():
    new_key = list(f"{str(key)}_ ")
    dict_matrix.append([new_key, value])

  iteration = True
  num_rounds = 0
  while iteration:
    token_freq = defaultdict(int) 
    for token_list, value in dict_matrix:
      for i in range(len(token_list)-2):
        # search key is current and next token
        search_key = token_list[i] + token_list[i+1]
        # add to dictionary if not already there
        token_freq[search_key] += value

    # word_freqs: go through list of words counting each pair of tokens
    c = Counter(token_freq)
    sorted_token_freq = {key: value for key, value in sorted(
        c.items(), key=lambda item: item[1], reverse=True)}

    # find most frequent token not already in vocab_bpe
    for token in sorted_token_freq.keys():
      if token not in vocab_bpe:
        first_token = token
        break
      else:
        first_token = None  # fallback in case all tokens are already in vocab

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
                # merge tokens
                token_list = token_list[:j] + [merged_token] + token_list[j + 2:]
                # don't increment j — might be able to merge again
            else:
                j += 1
        dict_matrix[i][0] = token_list
    if k:
       k -= 1
       iteration = (k > 0)

    else:
       num_rounds += 1
       accuracy = performance(dictionary, vocab_bpe, 500)
       if accuracy > 70: 
          iteration = False

       if num_rounds > 1500:
          print("exceeded, accuracy: ", accuracy)
          iteration = False

  return vocab_bpe, sorted_token_freq, dict_matrix


def get_best_merges(dict_train, text_train, dict_valid,text_valid,min_k,max_k,step):

  ks = []
  n_grams = []
  perplexities = []

  for k in range (min_k,max_k,step):
    vocab_train, _, _ = bpe(dict_train,k)
    n_gram_train = to_byte_pair(text_train, vocab_train)
    n_gram_valid = to_byte_pair(text_valid, vocab_train)

    our_n_grams_valid = generate_n_grams(n_gram_train,4, vocab_train)
    n_gram_num = 0

    for n_gram in our_n_grams_valid:
      n_gram_num += 1
      print(f"N-gram split for {n_gram_num}-gram, k = {k}: {n_gram.split_text[:10]}")
      perplexity = n_gram.perplexity(n_gram_valid, n_gram.ndim)
      perplexities.append(perplexity)
      ks.append(k)
      n_grams.append(n_gram_num)

  top_indices = find_top_indices(perplexities, 3)
  
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