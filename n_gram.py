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

  def __init__(self, corpus, n, vocab_size=15000):
    self.ndim = n
    self.unigram_probs = self.get_unigram_probs(corpus)
    self.split_text = self.split_ngrams(corpus)
    self.n_gram_probs = self.calculate_n_gram_probs(self.split_text)
    self.vocab_size = vocab_size



  def get_unigram_probs(self, corpus):
    """
    Pass corpus (already to byte-pair) and get unigram probs.
    """
    c = Counter(corpus)
    total = sum(c.values())
    # add count of 1 for Laplace Smoothing
    unigram_probs = {token: count+1 / total+self.vocab_size for token, count in c.items()} # is that normalised the way we want it
    return unigram_probs


  def split_ngrams(self, corpus):
    """
    Pass corpus (already to byte-pair)
    Funtion that chunks corpus into n-grams (n tokens are chunked together)
    """
    # want to split this into the maximum possible length
    # why am I not passing the vocabulary?
    n = self.ndim
    split_text = []
    for i in range(len(corpus) - n + 1):
      
      split_text.append(corpus[i : i+n])

    return split_text


  def calculate_n_gram_probs(self, split_text):
    """
    Builds conditional probabilities: P(w_n | w_1, ..., w_{n-1})
    Currently uses nested dictionaries, need to make that more useful for you
    """
    n = self.ndim
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
      # add count of 1 for Laplace Smoothing
      n_gram_probs[prefix] = {token: count +1/ total+self.vocab_size for token, count in target_counts.items()}

    return n_gram_probs

  def perplexity(self, split_text):
    """
    Calculate perplexity of a given text based on an n-gram model.

    Args:
        split_text (list): Tokenized input text (list of tokens).
        n_gram_probs (dict of dict): Nested dictionary with n-gram probabilities.
    Returns:
        float: Perplexity value of the input text.
    """
    log_prob_sum = 0.0
    count = 0

    # Iterate through text
    for t in range(len(split_text) - self.ndim + 1):
      context = tuple(split_text[t:t + self.ndim - 1])
      next_token = split_text[t + self.ndim - 1]

      # Small probability for unknown n-grams (smoothing)
      prob = self.n_gram_probs.get(context, {}).get(next_token, 1e-8)
      log_prob_sum += math.log(prob)
      count += 1

    # Calculate total probability in log space to avoid floating point underflow
    avg_log_prob = log_prob_sum / count
    # Exponentiate at the end to get the real perplexity.
    perplexity = math.exp(-avg_log_prob)

    print(f"Perplexity: {perplexity}")
    return perplexity