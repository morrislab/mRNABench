import pytest

pytest.importorskip("torch")

from mrna_bench.models.nucleotide_transformer import NucleotideTransformer


def token_count(num_nucleotides: int) -> int:
    """Tokens NT emits for a chunk: 6-mers, remainder, and CLS.

    Mirrors the tokenizer's behaviour of falling back to
    single-character tokens for nucleotides left over after the last
    complete 6-mer.
    """
    return 1 + num_nucleotides // 6 + num_nucleotides % 6


@pytest.mark.parametrize("max_length", [512, 1000, 2048])
def test_full_chunk_fits(max_length):
    """Check a maximally sized chunk stays within the token budget."""
    chunk = NucleotideTransformer.max_chunk_nucleotides(max_length)

    assert token_count(chunk) <= max_length


@pytest.mark.parametrize("max_length", [512, 1000, 2048])
def test_every_shorter_chunk_fits(max_length):
    """Check a trailing chunk of any length stays within the budget.

    A sequence's final chunk carries an arbitrary remainder, so every
    length up to the chunk size must fit, not just multiples of six.
    """
    chunk = NucleotideTransformer.max_chunk_nucleotides(max_length)
    worst = max(token_count(n) for n in range(chunk + 1))

    assert worst <= max_length


def test_rejects_the_previous_off_by_remainder_sizing():
    """Check the old (max_length - 2) * 6 sizing would have overflowed.

    Guards the regression directly: that sizing assumed a chunk costs
    L // 6 + 2 tokens, which holds only when L is a multiple of six.
    """
    max_length = 1000
    old_chunk = (max_length - 2) * 6

    assert token_count(old_chunk - 7) > max_length
    assert NucleotideTransformer.max_chunk_nucleotides(max_length) < old_chunk


def test_chunk_is_a_whole_number_of_kmers():
    """Check chunks split on 6-mer boundaries."""
    assert NucleotideTransformer.max_chunk_nucleotides(1000) % 6 == 0
