"""N-gram class

contains n-gram class with methods:
- get_unigram_probs: calculates unigram probabilities from a corpus
- split_ngrams: splits corpus into n-grams of specified length
- calculate_n_gram_probs: calculates conditional probabilities for n-grams
- perplexity: calculates perplexity of a given text based on n-gram probabilities

"""
from collections import Counter, defaultdict
import math
import random

class N_gram:

  def __init__(self, corpus, n, vocab):
    self.vocab = vocab
    self.ndim = n
    self.vocab_size = len(vocab)
    self.unigram_probs, self.counter = self.get_unigram_probs(corpus)
    self.split_text = self.split_ngrams(corpus)
    self.n_gram_probs = self.calculate_n_gram_probs(self.split_text)
    self.floor = 1e-8


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
    split_text = []
    for n in range(1, self.ndim+1):
      text = []
      for i in range(len(corpus) - n + 1):
        text.append(corpus[i : i+n])
      split_text.append(text)

    return split_text


  def calculate_n_gram_probs(self, split_text):
    """
    Builds conditional probabilities:, P(w_n | w_1, ..., w_{n-1}),
    using nested dictionaries.
    """
    n = self.ndim
    n_gram_probs = []
    for i in range(n):
      if i == 0:
        n_gram_probs.append(self.unigram_probs) 
      else:
        # nested defaultdicts for automatic initialization
        n_gram_counts = defaultdict(lambda: defaultdict(int))

        for n_gram in split_text[i]:
          
          prefix = tuple(n_gram[:-1])  # context: first n-1 tokens
          target = n_gram[-1]          # prediction target: nth token
          n_gram_counts[prefix][target] += 1

        # convert counts to probabilities
        n_g_probs = {}

        for prefix, target_counts in n_gram_counts.items():
          total = float(sum(target_counts.values()))
          n_g_probs[prefix] = {}
          for token in self.vocab:
              count = target_counts.get(token, 0)
              n_g_probs[prefix][token] = (count + 1) / (total + self.vocab_size)
        n_gram_probs.append(n_g_probs)
    return n_gram_probs
    
  def backoff_prob(self, context, token, n, generate = False):
    """
    Recursively back off from the full (n-1)-gram context down
    to unigram.  This mirrors exactly your `backoff(...)` in generate(),
    except it returns a probability instead of a sampled token.
    """
    # 1) If we're at the top order (len(context)==n-1), try to use it
    if n > 1:
      context = tuple(context)
      dist = self.n_gram_probs[n-1].get(context, None)

      if dist is not None:
        # seen this context
        #print(n)
        if generate: return dist
        return dist.get(token, 1/(self.counter[token]+self.vocab_size))    # if token unseen under this context, prob=0 here
      # 2) If we can still back off (i.e. n>1), drop the first item in context
      if len(context) > 0:
        return self.backoff_prob(context[1:], token, n-1, generate)
      
      # 3) Finally back off to unigram
    if generate: return self.unigram_probs
    return self.unigram_probs.get(token, self.floor)

  def perplexity(self, text, n):
    """
    Compute PP = exp{-1/M ∑_i log P(w_i | context_i)} using
    exactly the same backoff-probabilities as generate().
    """
    log_prob = 0.0
    M = 0
    floor = self.floor
    for i in range(len(text) - n + 1):

      ctx = text[i : i + n - 1]
      w   = text[i + n - 1]
      p = self.backoff_prob(ctx, w, n)
      # if p is zero (never seen at any order, even unigram), floor it  
      if p == 0.0:
        p = floor
      log_prob += math.log(p)
      M += 1
    avg_ll = log_prob / M
    pp = math.exp(-avg_ll)
    print(f"Order: {n}, perplexity: {pp:.2f}")
    return pp
  
  def generate(self, n, max_length=100, seed=None, k = 5):
    """
    Generate a sequence of tokens using the n-gram model with backoff.
    """
    end_tokens = ['.', ':', '?', '!', '._', '!_', '?_', ':_', ';_']

    if seed is None:
      current_token = random.choice(self.vocab)
      output = [current_token]
    else:
      output = seed

    for _ in range(max_length - 1):
      context = output[-(n - 1):] 
      next_token = self.backoff_prob(context, None, n, generate=True)
      # Sample randomly from the top k tokens by probability
      sorted_tokens = sorted(next_token.items(), key=lambda item: item[1], reverse=True)
      top_k = sorted_tokens[:k]
      tokens, probs = zip(*top_k)    
      next_token = random.choices(tokens, weights=probs, k=1)[0]
      output.append(next_token)
      if next_token in end_tokens:
        break
    
    pretty_text = "".join(str(x) for x in output).replace('_', ' ')

    return pretty_text