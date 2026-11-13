"""Train a byte-level BPE tokenizer."""

from collections import Counter, defaultdict

import regex as re


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def initialize_vocab(special_tokens: list[str]) -> dict[int, bytes]:
    i = 0
    vocab = {}
    while i < 256:
        vocab[i] = bytes([i])
        i = i + 1
    for special_token in special_tokens:
        vocab[i] = special_token.encode("utf-8")
        i = i + 1
    return vocab


def split_on_special_tokens(texts: str, special_tokens: list[str]) -> list[str]:
    escapes = []
    text_segments = [texts]
    for special_token in special_tokens:
        escapes.append(re.escape(special_token))
    if len(escapes) > 0:
        text_segments = re.split("|".join(escapes), texts)
    return text_segments


def count_pretokens(text_segments: list[str]) -> dict[tuple[bytes, ...], int]:
    """Count UTF-8 byte sequences without crossing special-token boundaries."""
    counts = {}
    for segment in text_segments:
        for match in re.finditer(PAT, segment):
            encoded = match.group(0).encode("utf-8")
            pretoken = tuple(bytes([value]) for value in encoded)
            counts[pretoken] = counts.get(pretoken, 0) + 1
    return counts


Pair = tuple[bytes, bytes]
Pretoken = tuple[bytes, ...]


def _build_pair_index(
    pretokens: list[tuple[Pretoken, int]],
) -> tuple[Counter[Pair], dict[Pair, set[int]]]:
    pair_counts = Counter()
    pair_locations = defaultdict(set)
    for index, (pretoken, frequency) in enumerate(pretokens):
        for pair, occurrences in Counter(zip(pretoken, pretoken[1:])).items():
            pair_counts[pair] += occurrences * frequency
            pair_locations[pair].add(index)
    return pair_counts, pair_locations


def _merge_pretoken(pretoken: Pretoken, pair: Pair, merged_token: bytes) -> Pretoken:
    left, right = pair
    merged_tokens = []
    i = 0
    while i < len(pretoken) - 1:
        if pretoken[i] == left and pretoken[i + 1] == right:
            merged_tokens.append(merged_token)
            i += 2
        else:
            merged_tokens.append(pretoken[i])
            i += 1
    return tuple(merged_tokens) + pretoken[i:]


def _apply_merge(
    pretokens: list[tuple[Pretoken, int]],
    pair_counts: Counter[Pair],
    pair_locations: dict[Pair, set[int]],
    top_pair: Pair,
    merged_token: bytes,
) -> None:
    # The index changes as each affected pretoken is replaced.
    for index in tuple(pair_locations[top_pair]):
        pretoken, frequency = pretokens[index]
        new_pretoken = _merge_pretoken(pretoken, top_pair, merged_token)
        pretokens[index] = (new_pretoken, frequency)

        old_pairs = Counter(zip(pretoken, pretoken[1:]))
        new_pairs = Counter(zip(new_pretoken, new_pretoken[1:]))
        for pair in old_pairs.keys() | new_pairs.keys():
            pair_counts[pair] += (new_pairs[pair] - old_pairs[pair]) * frequency
            if not pair_counts[pair]:
                del pair_counts[pair]
                del pair_locations[pair]
            elif pair not in new_pairs:
                pair_locations[pair].remove(index)
            else:
                pair_locations[pair].add(index)


def train_bpe(
    input_path: str, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[Pair]]:
    with open(input_path, encoding="utf-8") as file:
        texts = file.read()
    text_segments = split_on_special_tokens(texts, special_tokens)
    pretokens = list(count_pretokens(text_segments).items())
    vocab = initialize_vocab(special_tokens)
    merges = []
    pair_counts, pair_locations = _build_pair_index(pretokens)

    while len(vocab) < vocab_size and pair_counts:
        top_pair = max(pair_counts, key=lambda pair: (pair_counts[pair], pair))
        merged_token = b"".join(top_pair)
        vocab[len(vocab)] = merged_token
        merges.append(top_pair)
        _apply_merge(pretokens, pair_counts, pair_locations, top_pair, merged_token)
    return vocab, merges
