from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Protocol

import numpy as np
import torch
from torch.utils.data import IterableDataset

PAD_ID = 3


class TokenizerProtocol(Protocol):
    def encode(self, text: str) -> list[int]:
        ...


def _sha256_file(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _token_cache_path(text_path: Path, cache_dir: Path) -> Path:
    file_hash = _sha256_file(text_path)
    return cache_dir / f"{text_path.stem}.{file_hash}.npy"


def _tokenize_file(
    text_path: Path, tokenizer: TokenizerProtocol, cache_dir: Path, dtype: np.dtype
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    token_path = _token_cache_path(text_path, cache_dir)
    if token_path.exists():
        return token_path
    text = text_path.read_text(encoding="utf-8")
    tokens = tokenizer.encode(text)
    tokens_np = np.asarray(tokens, dtype=dtype)
    np.save(token_path, tokens_np, allow_pickle=False)
    return token_path


@dataclass(frozen=True)
class IndexEntry:
    file_path: str
    token_offset: int


class DocumentDataset(IterableDataset):
    def __init__(
        self,
        file_paths: Iterable[str | Path],
        tokenizer: TokenizerProtocol,
        seq_len: int,
        cache_dir: str | Path,
        seed: int = 0,
        shuffle: bool = True,
        dtype: np.dtype = np.uint32,
        pad_id: int = PAD_ID,
    ) -> None:
        super().__init__()
        if seq_len <= 0:
            raise ValueError("seq_len must be positive.")
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.chunk_len = seq_len + 1
        self.cache_dir = Path(cache_dir)
        self.seed = seed
        self.shuffle = shuffle
        self.dtype = dtype
        self.pad_id = pad_id
        self._memmap_cache: dict[str, np.ndarray] = {}

        self.file_paths = [Path(p) for p in file_paths]
        self.file_paths.sort()
        if not self.file_paths:
            raise ValueError("file_paths must contain at least one document.")

        self._token_paths: dict[str, Path] = {}
        self._index: list[IndexEntry] = self._build_index()

    def _build_index(self) -> list[IndexEntry]:
        index: list[IndexEntry] = []
        for path in self.file_paths:
            token_path = _tokenize_file(path, self.tokenizer, self.cache_dir, self.dtype)
            self._token_paths[str(path)] = token_path
            tokens = np.load(token_path, mmap_mode="r")
            n_tokens = int(tokens.shape[0])
            if isinstance(tokens, np.memmap) and tokens._mmap is not None:
                tokens._mmap.close()
            n_chunks = n_tokens // self.chunk_len
            for i in range(n_chunks):
                offset = i * self.chunk_len
                index.append(IndexEntry(str(path), offset))
        if self.shuffle:
            rng = random.Random(self.seed)
            rng.shuffle(index)
        return index

    def _load_tokens(self, file_path: str) -> np.ndarray:
        token_path = self._token_paths[file_path]
        token_path_str = str(token_path)
        tokens = self._memmap_cache.get(token_path_str)
        if tokens is None:
            tokens = np.load(token_path, mmap_mode="r")
            self._memmap_cache[token_path_str] = tokens
        return tokens

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        for idx, entry in enumerate(self._index):
            print(f"[DEBUG] __iter__ index={idx}, file={Path(entry.file_path).name}, offset={entry.token_offset}")
            tokens = self._load_tokens(entry.file_path)
            start = entry.token_offset
            end = start + self.chunk_len
            chunk = tokens[start:end]
            if int(chunk.shape[0]) != self.chunk_len:
                raise RuntimeError("Index out of range for token chunk.")
            # FIX C: Create a writable copy to avoid PyTorch warning about non-writable arrays
            chunk = np.array(chunk, dtype=np.int64, copy=True)
            input_ids = torch.from_numpy(chunk[:-1])
            labels = torch.from_numpy(chunk[1:])
            yield input_ids, labels

    def __len__(self) -> int:
        return len(self._index)

    def close(self) -> None:
        # FIX B: Explicitly close all memory-mapped files (critical for Windows cleanup)
        for tokens in self._memmap_cache.values():
            if isinstance(tokens, np.memmap) and tokens._mmap is not None:
                tokens._mmap.close()
        self._memmap_cache.clear()


def generate_synthetic_corpus(
    root_dir: str | Path, num_docs: int, tokens_per_doc: int
) -> list[Path]:
    root = Path(root_dir)
    root.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for i in range(num_docs):
        token = f"DOC{i}"
        text = " ".join([token] * tokens_per_doc)
        path = root / f"doc_{i:04d}.txt"
        path.write_text(text, encoding="utf-8")
        paths.append(path)
    return paths
