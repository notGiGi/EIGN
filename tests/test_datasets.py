from pathlib import Path

import torch

from eign.data.datasets import DocumentDataset, generate_synthetic_corpus


class DummyTokenizer:
    def __init__(self, vocab: list[str]) -> None:
        self.vocab = {token: i + 1 for i, token in enumerate(vocab)}

    def encode(self, text: str) -> list[int]:
        tokens = text.strip().split()
        return [self.vocab[tok] for tok in tokens if tok]


def test_shapes_and_shift(tmp_path: Path) -> None:
    doc_dir = tmp_path / "docs"
    doc_dir.mkdir(parents=True, exist_ok=True)
    doc_path = doc_dir / "doc0.txt"
    doc_path.write_text("A B C D E F", encoding="utf-8")
    tokenizer = DummyTokenizer(["A", "B", "C", "D", "E", "F"])

    dataset = DocumentDataset(
        [doc_path], tokenizer, seq_len=3, cache_dir=tmp_path / "cache", shuffle=False
    )
    input_ids, labels = next(iter(dataset))

    assert input_ids.shape == (3,)
    assert labels.shape == (3,)
    assert torch.equal(labels[:-1], input_ids[1:])


def test_deterministic_ordering(tmp_path: Path) -> None:
    doc_dir = tmp_path / "docs"
    paths = generate_synthetic_corpus(doc_dir, num_docs=3, tokens_per_doc=9)
    tokenizer = DummyTokenizer(["DOC0", "DOC1", "DOC2"])

    dataset_a = DocumentDataset(
        paths,
        tokenizer,
        seq_len=3,
        cache_dir=tmp_path / "cache",
        seed=123,
        shuffle=True,
    )
    dataset_b = DocumentDataset(
        paths,
        tokenizer,
        seq_len=3,
        cache_dir=tmp_path / "cache",
        seed=123,
        shuffle=True,
    )

    samples_a = [tuple(x.tolist()) for x, _ in dataset_a]
    samples_b = [tuple(x.tolist()) for x, _ in dataset_b]
    assert samples_a == samples_b


def test_no_cross_document_leakage(tmp_path: Path) -> None:
    doc_dir = tmp_path / "docs"
    paths = generate_synthetic_corpus(doc_dir, num_docs=2, tokens_per_doc=8)
    tokenizer = DummyTokenizer(["DOC0", "DOC1"])

    dataset = DocumentDataset(
        paths,
        tokenizer,
        seq_len=3,
        cache_dir=tmp_path / "cache",
        seed=0,
        shuffle=False,
    )

    for input_ids, labels in dataset:
        assert torch.unique(input_ids).numel() == 1
        assert torch.unique(labels).numel() == 1
        assert input_ids[0].item() == labels[0].item()
