# Imports
import numpy as np
from typing import List, Tuple
from numpy.typing import ArrayLike

def sample_seqs(seqs: List[str], labels: List[bool]) -> Tuple[List[str], List[bool]]:
    """
    This function should sample the given sequences to account for class imbalance. 
    Consider this a sampling scheme with replacement.
    
    Args:
        seqs: List[str]
            List of all sequences.
        labels: List[bool]
            List of positive/negative labels

    Returns:
        sampled_seqs: List[str]
            List of sampled sequences which reflect a balanced class size
        sampled_labels: List[bool]
            List of labels for the sampled sequences
    """
    # Separate sequences by label
    pos_seqs = [s for s, l in zip(seqs, labels) if l]
    neg_seqs = [s for s, l in zip(seqs, labels) if not l]

    n_pos = len(pos_seqs)
    n_neg = len(neg_seqs)

    # Upsample the minority class to match the majority class
    if n_pos < n_neg:
        sampled_pos = np.random.choice(pos_seqs, size=n_neg, replace=True).tolist()
        sampled_seqs = sampled_pos + neg_seqs
        sampled_labels = [True] * len(sampled_pos) + [False] * len(neg_seqs)
    elif n_neg < n_pos:
        sampled_neg = np.random.choice(neg_seqs, size=n_pos, replace=True).tolist()
        sampled_seqs = pos_seqs + sampled_neg
        sampled_labels = [True] * len(pos_seqs) + [False] * len(sampled_neg)
    else:
        sampled_seqs = seqs
        sampled_labels = labels

    return sampled_seqs, sampled_labels

def one_hot_encode_seqs(seq_arr: List[str]) -> ArrayLike:
    """
    This function generates a flattened one-hot encoding of a list of DNA sequences
    for use as input into a neural network.

    Args:
        seq_arr: List[str]
            List of sequences to encode.

    Returns:
        encodings: ArrayLike
            Array of encoded sequences, with each encoding 4x as long as the input sequence.
            For example, if we encode:
                A -> [1, 0, 0, 0]
                T -> [0, 1, 0, 0]
                C -> [0, 0, 1, 0]
                G -> [0, 0, 0, 1]
            Then, AGA -> [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0].
    """
    mapping = {
        'A': [1, 0, 0, 0],
        'T': [0, 1, 0, 0],
        'C': [0, 0, 1, 0],
        'G': [0, 0, 0, 1]
    }

    encodings = []
    for seq in seq_arr:
        encoded_seq = []
        for base in seq:
            if base in mapping:
                encoded_seq.extend(mapping[base])
            else:
                encoded_seq.extend([0, 0, 0, 0])
        encodings.append(encoded_seq)

    return np.array(encodings)