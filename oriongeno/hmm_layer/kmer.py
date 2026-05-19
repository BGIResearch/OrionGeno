import torch

def make_k_mers(sequences, k, pivot_left=True):
    """Convert one-hot nucleotide sequences into k-mer emission tensors."""
    L = sequences.shape[-2]
    n = sequences.shape[-1] - 1
    n = torch.tensor(n, dtype=sequences.dtype, device=sequences.device)

    sequences_no_N = sequences[..., :-1]
    N_pos = (sequences[..., -1:] == 1).to(sequences.dtype)
    sequences_no_N += (1 / n) * N_pos

    pad = torch.ones_like(sequences_no_N[:, :k - 1, :], dtype=sequences.dtype) / n
    if pivot_left:
        sequences_padded_no_N = torch.cat([sequences_no_N, pad], dim=-2)
        k_mers = sequences_padded_no_N[:, :L, None, :]
    else:
        sequences_padded_no_N = torch.cat([pad, sequences_no_N], dim=-2)
        k_mers = sequences_padded_no_N[:, k - 1:L + k - 1, None, :]
    if pivot_left:
        iteration_range = range(1, k)
    else:
        iteration_range = range(k - 2, -1, -1)

    for i in iteration_range:
        shift_i = sequences_padded_no_N[:, i:L + i, None, :, None]
        k_mers = k_mers[..., None, :] * shift_i
        if pivot_left:
            shape = [4**i, 4]
        else:
            shape = [4**(k - i - 1), 4]
        k_mers = k_mers.view(list(k_mers.shape[:-3]) + shape)
    return k_mers

def encode_kmer_string(kmer, pivot_left=True, alphabet="ACGT"):
    """Encode a literal k-mer string into the k-mer class layout."""
    alphabet_with_unknown = alphabet + "N"
    kmer = [alphabet_with_unknown.index(x) for x in kmer]
    kmer = torch.tensor(kmer)
    one_hot = torch.nn.functional.one_hot(kmer, num_classes=len(alphabet_with_unknown)).to(torch.float32)
    encoded_kmers = make_k_mers(one_hot.unsqueeze(0), k=len(kmer), pivot_left=pivot_left)
    if pivot_left:
        return encoded_kmers.squeeze(0)[0]
    else:
        return encoded_kmers.squeeze(0)[-1]
