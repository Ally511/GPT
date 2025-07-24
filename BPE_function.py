"""BPE function

function to create byte pairs based off a dictionary of tokens and their counts.
"""
import numpy as np
from collections import defaultdict
from collections import OrderedDict
from collections import Counter

from utility_functions import performance

def bpe(dictionary, k=None):
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


  token_freq = defaultdict(int) # TODO: does that need to be moved outside of the loop?
  iteration = True
  num_rounds = 0
  while iteration:

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

    # höchster count wird gemerged:
    # add to vocab
    #first_token = list(sorted_token_freq.keys())[0]
    #vocab_bpe.append(first_token)
    # replace in list of words
    # start again?
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
       iteration != (accuracy > 70)

       if num_rounds > 1500:
          print("exceeded, accuracy: ", accuracy)


          iteration = False

  return vocab_bpe, sorted_token_freq, dict_matrix