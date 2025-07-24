"""N-gram class

contains n-gram class with methods:
- get_unigram_probs: calculates unigram probabilities from a corpus
- split_ngrams: splits corpus into n-grams of specified length
- calculate_n_gram_probs: calculates conditional probabilities for n-grams
- perplexity: calculates perplexity of a given text based on n-gram probabilities

"""
from collections import Counter, defaultdict
import math

class N_gram:

  def __init__(self, corpus, n):
    self.ndim = n
    self.unigram_probs = self.get_unigram_probs(corpus)
    self.split_text = self.split_ngrams(corpus, n)
    self.n_gram_probs = self.calculate_n_gram_probs(self.split_text, n)



  def get_unigram_probs(self, corpus):
    """
    Pass corpus (already to byte-pair) and get unigram probs.
    """
    c = Counter(corpus)
    total = sum(c.values())
    unigram_probs = {token: count / total for token, count in c.items()} # is that normalised the way we want it
    return unigram_probs


  def split_ngrams(self, corpus, n):
    """
    Pass corpus (already to byte-pair)
    Funtion that chunks corpus into n-grams (n tokens are chunked together)
    """
    # want to split this into the maximum possible length
    # why am I not passing the vocabulary?
    print(type(corpus))
    split_text = []
    for i in range(len(corpus) - n + 1):
      
      split_text.append(corpus[i : i+n])

    return split_text


  def calculate_n_gram_probs(self, split_text, n):
    """
    Builds conditional probabilities: P(w_n | w_1, ..., w_{n-1})
    Currently uses nested dictionaries, need to make that more useful for you
    """
    if n == 1:
      return self.unigram_probs
    
    # nested defaultdicts for automatic initialization
    n_gram_counts = defaultdict(lambda: defaultdict(int))

    for n_gram in split_text:
      prefix = tuple(n_gram[:-1])  # context: first n-1 tokens
      target = n_gram[-1]          # prediction target: nth token
      n_gram_counts[prefix][target] += 1

    # convert counts to probabilities
    n_gram_probs = {}

    for prefix, target_counts in n_gram_counts.items():
      total = float(sum(target_counts.values()))
      n_gram_probs[prefix] = {token: count / total for token, count in target_counts.items()}


    return n_gram_probs

  def perplexity(self, split_text, n_gram_probs):
    """
    Calculate perplexity of a given text based on an n-gram model.

    Args:
        split_text (str): Input text string, use test text.
        n_gram_probs (dict of dict): ...
    Returns:
        float: Perplexity value of the input text.
    """

    #text = generator.to_byte_pair.to_byte_pair(text,vocab) # already done
    #n = self.ndim  # order of the n-gram model # can access through class
    prob_total = 1.0

    # Iterate through the text, considering n-grams
    for t in range(len(split_text) - self.ndim):
      context = tuple(split_text[t:t + self.ndim - 1])  # (n-1)-gram
      next_token = split_text[t + self.ndim - 1]        # next token after context

      # Assign very small probability for unknown n-grams (smoothing) otherwise retrieve actual probability
      prob = 0.0
      if context in n_gram_probs and next_token in n_gram_probs[context]:
        prob = n_gram_probs[context][next_token]
      else:
        prob = 1e-10  # unknown n-gram

      prob_total *= prob

    print(f"Total probability: {prob_total}")
    if prob_total > 0.0:
      perplexity_value = prob_total ** (-1 / len(split_text))
    else:
      perplexity_value = float('inf')

    print(f"Perplexity value: {perplexity_value}")
    return perplexity_value

