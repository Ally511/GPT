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

def plot_prplx_diff(all_perplexities, fig_path):
    n_grams = [1, 2, 3, 4]
    versions = ["2nd", "3rd"]

    # compute differences relative to "Best"
    diffs = all_perplexities[1:] - all_perplexities[0]

    plt.figure(figsize=(8,5))
    for i in range(diffs.shape[0]):
        plt.plot(n_grams, diffs[i], marker='o', label=versions[i])

    plt.xticks(n_grams)
    plt.xlabel("N-gram Order")
    plt.ylabel("Δ Perplexity (vs. Best)")
    plt.title("Perplexity Difference Relative to Best Merge")
    plt.axhline(0, color="black", linewidth=1, linestyle="--")
    plt.legend()
    plt.grid(True)
    plt.savefig(fig_path)

"""
def extrinsic_eval_all(all_n_grams, vocab, out_path=None):
    print("Starting extrinsic eval")
    n_out_all = []
    for n_gram_list in all_n_grams:
        print("Next merge")
        n_list = []
        n_out_set = []
        for n_gram in n_gram_list:
            n_list.append(n_gram.n_gram_probs)
            print(len(n_list))
            n_gram_out = generate("cleopatra is my", n_list, 2, vocab)
            # save text to file 
            n_out_set.append(n_gram_out)
            
            #n_gram_out = ("\n N-gram of order: ", n_gram.ndim, n_gram_out)
            #with open(out_path, "a") as f:
                #f.write(str(n_gram_out))
        n_out_all.append(np.array(n_gram_out))
    return np.array(n_out_all)"""

"""import pandas as pd

def extrinsic_eval_all(all_n_grams, vocab, out_path=None):
    # don't think the Unigram can generate, fix that
    print("Starting extrinsic eval")
    n_out_all = []  # will hold lists of outputs per dataset version

    for model_idx, n_gram_list in enumerate(all_n_grams):
        print(f"Next merge (model {model_idx+1})")
        n_out_set = []
        n_list = [n_gram_list[0].n_gram_probs]
        
        # generate a sample from each n-gram order
        for n_gram in n_gram_list[1:]:
            n_list.append(n_gram.n_gram_probs)
            n_gram_out = generate("cleopatra is my", n_list, 2, vocab)
            n_out_set.append(n_gram_out)
        
        n_out_all.append(n_out_set)

    # convert to dataframe for easy table export
    df = pd.DataFrame(n_out_all, columns=["Bigram", "Trigram", "4-gram"])
    df.index = ["Best", "2nd", "3rd"]

    # optionally save as markdown
    if out_path:
        with open(out_path, "w") as f:
            f.write(df.to_markdown())

    return df"""

def extrinsic_eval_all(all_n_grams, vocab, context="cleopatra is my", out_path=None):
    # all_n_grams: list of lists of N_gram objects for each dataset version
    all_outputs = []
    for n_gram_list in all_n_grams:
        # convert objects → dicts expected by generate()
        ngram_dicts = [ng.n_gram_probs for ng in n_gram_list]
        outputs = []
        for order in range(1, len(ngram_dicts)+1):
            outputs.append(generate(context, ngram_dicts, order, vocab))
        all_outputs.append(outputs)

    # Optional: to a markdown table
    try:
        import pandas as pd
        df = pd.DataFrame(
            all_outputs,
            columns=["Unigram", "Bigram", "Trigram", "4-gram"],
            index=[f"Version {i+1}" for i in range(len(all_outputs))]
        )
        if out_path:
            df.to_markdown(out_path)
        else:
            print(df.to_markdown())
    except Exception:
        pass

    return all_outputs



def extrinsic_eval_all_prob_fine(all_n_grams, vocab, context="cleopatra is my", out_path=None):
    """
    Run extrinsic evaluation on multiple models/versions and return generated samples.

    Args:
        all_n_grams: list of lists, each inner list = [unigram_model, bigram_model, ...]
        vocab: list of subword tokens
        context: str, starting context for generation
        out_path: optional path to save results as CSV/Markdown

    Returns:
        results: list of lists of strings
            results[i][j] = sample from model i, n-gram order j
    """
    print("Starting extrinsic eval")
    all_outputs = []

    for model_idx, n_gram_list in enumerate(all_n_grams):
        print(f"Next dataset version: {model_idx+1}")
        outputs_per_model = []
        n_list = [ng.n_gram_probs for ng in n_gram_list]  # dicts only
        for n in range(1, len(n_list)+1):
            print(f"  Generating with {n}-gram")
            n_gram_out = generate(context, n_list, n, vocab)
            outputs_per_model.append(n_gram_out)
        
        all_outputs.append(outputs_per_model)

    # Optional: save as markdown table
    if out_path:
        import pandas as pd
        df = pd.DataFrame(
            all_outputs,
            columns=["Unigram", "Bigram", "Trigram", "4-gram"],
            index=[f"Version {i+1}" for i in range(len(all_outputs))]
        )
        df.to_markdown(out_path)

    return all_outputs


with open (r"corpora/vocab_train.txt", 'r') as f:
  vocab_train = eval(f.read())
all_train = ("Shakespeare_best_merge_train.txt", "Shakespeare_2nd_best_merge_train.txt", "Shakespeare_3rd_best_merge_train.txt")
all_test = ("Shakespeare_best_merge_test.txt", "Shakespeare_2nd_best_merge_test.txt", "Shakespeare_3rd_best_merge_test.txt")

all_perplexities, all_n_grams = intrinsic_eval_all(all_train, all_test, vocab_train)
print(all_perplexities)
"""fig_path_1 = Path('img') / 'n_gram_perplexities.png'
fig_path_2 = Path('img') / 'n_gram_prplx_diff.png'
#plot_perplexities(all_perplexities, fig_path_1)
plot_prplx_diff(all_perplexities, fig_path_2)
out_path_df = Path('n_gram') / 'n_gram_sample.md'
all_text_output = extrinsic_eval_all(all_n_grams, vocab_train, context="cleopatra is my", out_path=out_path_df)
print(all_text_output)"""

# 1) Does your initial context exist at higher orders?
tokens = to_byte_pair("cleopatra is my", vocab_train)
for m, d in enumerate([ng.n_gram_probs for ng in all_n_grams[0]], start=1):
    key = tuple(tokens[-(m-1):]) if m > 1 else None
    if m == 1:
        print("Unigram has", len(d), "entries")
    else:
        print(f"{m}-gram has context {key!r}? ->", key in d)

# 2) Verify that some trigram contexts have continuations
tri = all_n_grams[0][3-1].n_gram_probs  # trigram dict
print("Example trigram contexts:", list(tri)[:3])



"""n_list = [all_n_grams[0][0].n_gram_probs, all_n_grams[0][1].n_gram_probs,all_n_grams[0][2].n_gram_probs]

text = generate("cleopatra is my", n_list, 2, vocab_train)
print(text)"""
# need to add out_path before I can get 

