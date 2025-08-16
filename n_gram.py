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

  def __init__(self, corpus, n, vocab_size):
    self.ndim = n
    self.vocab_size = vocab_size
    self.unigram_probs, self.counter = self.get_unigram_probs(corpus)
    self.split_text = self.split_ngrams(corpus)
    self.n_gram_probs = self.calculate_n_gram_probs(self.split_text)


  def get_unigram_probs(self, corpus):
    """
    Pass corpus (already to byte-pair) and get unigram probs.
    """
    c = Counter(corpus)
    total = sum(c.values())
    # add count of 1 for Laplace Smoothing
    unigram_probs = {token: (count+1) / (total+self.vocab_size) for token, count in c.items()} # is that normalised the way we want it
    return unigram_probs, c


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
      n_gram_probs[prefix] = {token: (count +1)/ (total+self.vocab_size) for token, count in target_counts.items()}

    return n_gram_probs
    
  def backoff_prob(self, context, token):
    """
    Recursively back off from the full (n-1)-gram context down
    to unigram.  This mirrors exactly your `backoff(...)` in generate(),
    except it returns a probability instead of a sampled token.
    """
    # 1) If we're at the top order (len(context)==n-1), try to use it
    if len(context) == self.ndim - 1:
      dist = self.n_gram_probs.get(tuple(context))
      if dist is not None:
        # seen this context
        return dist.get(token, 1/(self.counter[token]+self.vocab_size))    # if token unseen under this context, prob=0 here

    # 2) If we can still back off (i.e. n>1), drop the first item in context
    if len(context) > 0:
      return self._backoff_prob(context[1:], token)

    # 3) Finally back off to unigram
    return self.unigram_probs.get(token, self.floor)

  def perplexity(self, split_text, floor = 1e-8):
    """
    Compute PP = exp{-1/M ∑_i log P(w_i | context_i)} using
    exactly the same backoff-probabilities as generate().
    """
    log_prob = 0.0
    M = 0
    n = self.ndim

    for i in range(len(split_text) - n + 1):
      ctx = split_text[i : i + n - 1]
      w   = split_text[i + n - 1]

      p = self.backoff_prob(ctx, w)
      # if p is zero (never seen at any order, even unigram), floor it  
      if p == 0.0:
        p = floor

      log_prob += math.log(p)
      M += 1

    avg_ll = log_prob / M
    pp = math.exp(-avg_ll)
    print(f"Perplexity: {pp:.2f}")
    return pp

  def old_perplexity(self, split_text):
    
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
    #dividing by amount of n-gram predictions made, could also divide by len(tokens)
    avg_log_prob = log_prob_sum / count
    # Exponentiate at the end to get the real perplexity.
    perplexity = math.exp(-avg_log_prob)

    print(f"Old perplexity: {perplexity}")
    return perplexity