from __future__ import annotations

from pathlib import Path

import sentencepiece as spm


class SentencePieceTokenizer:
    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path)
        self.sp = spm.SentencePieceProcessor(model_file=str(self.model_path))
        self.unk_id = self.sp.unk_id()
        self.bos_id = self.sp.bos_id()
        self.eos_id = self.sp.eos_id()
        self.pad_id = self.sp.pad_id()

    def encode(self, text: str) -> list[int]:
        return self.sp.encode(text, out_type=int)

    def decode(self, ids: list[int]) -> str:
        return self.sp.decode(ids)

    @property
    def vocab_size(self) -> int:
        return self.sp.vocab_size()


def train_tokenizer(
    input_files: list[Path],
    model_prefix: str,
    vocab_size: int = 32000,
    model_type: str = "unigram",
    character_coverage: float = 0.9995,
    pad_id: int = 3,
) -> Path:
    """Train SentencePiece tokenizer from text files.

    Args:
        input_files: List of text files to train on
        model_prefix: Output model prefix (e.g., 'path/to/tokenizer')
        vocab_size: Target vocabulary size
        model_type: SentencePiece model type (unigram, bpe, char, word)
        character_coverage: Character coverage for tokenization
        pad_id: Padding token ID

    Returns:
        Path to trained .model file
    """
    model_prefix_path = Path(model_prefix)
    model_prefix_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert input files to comma-separated string
    input_str = ",".join(str(f) for f in input_files)

    # Train tokenizer
    spm.SentencePieceTrainer.train(
        input=input_str,
        model_prefix=str(model_prefix_path),
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=character_coverage,
        pad_id=pad_id,
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_piece="<pad>",
        unk_piece="<unk>",
        bos_piece="<s>",
        eos_piece="</s>",
        normalization_rule_name="identity",
        remove_extra_whitespaces=False,
        max_sentence_length=16384,
        shuffle_input_sentence=True,
        seed_sentencepiece_size=1000000,
        train_extremely_large_corpus=False,
    )

    model_file = model_prefix_path.with_suffix(".model")
    return model_file
