"""import self-written function and classes"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from n_gram.generator import to_byte_pair,generate
from n_gram.n_gram import N_gram
from utility_functions import get_top_bigrams,get_words
from utility_functions import generate_n_grams
from BPE_function import bpe,get_best_merges
# okay, i want a file, where, for each of the k splits, I generate the four n-grams
# then intrinsic evaluation: calculate AND PLOT perplexity
# --> save perplexity by data used and by n-gram to evaluate influence
# then run extrinsic evaluation for all three texts and all four n-grams
# save outputs ...somehow?


# instead of printing, save to pd data frame? 

def intrinsic_eval_set(n_gram_list, test_corpus):
    print("starting intrinsic eval for set")
    perplexities = []
    for n_gram in n_gram_list:
        print("N-gram of order: ", n_gram.ndim)
        perplexity = n_gram.perplexity(test_corpus)
        perplexities.append(perplexity)
    perplexities = np.array(perplexities)
    return perplexities

def intrinsic_eval_all(paths_train, paths_test, vocab):
    print("starting intrinsic evaluation")
    all_perplexities = []
    all_n_grams = []
    for path_train, path_test in zip(paths_train, paths_test):
        with open(path_train, "r") as f:
            n_gram_corps_train = f.read()
        with open(path_test, "r") as f:
            n_gram_corps_test = f.read()
        our_n_grams = generate_n_grams(n_gram_corps_train,4, len(vocab))
        # unigram always idx 0, bigram always idx 1, etc.
        perplexities = intrinsic_eval_set(our_n_grams, n_gram_corps_test)
        all_perplexities.append(perplexities)
        all_n_grams.append(our_n_grams)
    all_perplexities = np.array(all_perplexities)
    return all_perplexities, all_n_grams

def plot_perplexities(all_perplexities, fig_path):
    n_grams = [1, 2, 3, 4]
    versions = ["Best", "2nd", "3rd"]
    plt.figure(figsize=(8,5))
    for i in range(all_perplexities.shape[0]):
        plt.plot(n_grams, all_perplexities[i], marker='o', label=versions[i])
        plt.xticks(n_grams)
        plt.xlabel("N-gram Order")
        plt.ylabel("Perplexity")
        plt.title("N-gram Perplexity Across Dataset Versions")
        plt.legend()
        plt.grid(True)
        plt.savefig(fig_path)

def extrinsic_eval_all(all_n_grams, vocab, out_path):
    for n_gram_list in all_n_grams:
        n_list = []
        for n_gram in n_gram_list:
            n_list.append(n_gram.n_gram_probs)
            n_gram_out = generate("cleopatra is my", n_list, 2, vocab)
            # save text to file 
            n_gram_out = ("\n N-gram of order: ", n_gram.ndim, n_gram_out)
            with open(out_path, "a") as f:
                f.write(str(n_gram_out))



with open (r"corpora/vocab_train.txt", 'r') as f:
  vocab_train = eval(f.read())
all_train = ("Shakespeare_best_merge_train.txt", "Shakespeare_2nd_best_merge_train.txt", "Shakespeare_3rd_best_merge_train.txt")
all_test = ("Shakespeare_best_merge_test.txt", "Shakespeare_2nd_best_merge_test.txt", "Shakespeare_3rd_best_merge_test.txt")

all_perplexities, all_n_grams = intrinsic_eval_all(all_train, all_test, vocab_train)
print(all_perplexities)
fig_path = Path('img') / 'n_gram_perplexities.png'
plot_perplexities(all_perplexities, fig_path)
# need to add out_path before I can get 

