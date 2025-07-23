from itertools import islice

def performance(sorted_dict, vocab, k=10000):

    def take(n, iterable):
        """Return the first n items of the iterable as a list."""
        return list(islice(iterable, n))

    qualifier = take(k, sorted_dict.items())

    overlap = set(qualifier) & set(vocab)

    # Anzahl der Treffer
    num_overlap = len(overlap)

    # Prozent ausrechnen
    percentage = (num_overlap / len(qualifier)) * 100
    return percentage
