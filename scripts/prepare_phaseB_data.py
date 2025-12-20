#!/usr/bin/env python
"""
EIGN Phase B Data Preparation Script

Downloads and prepares clean plaintext datasets for continued pretraining:
- Wikipedia English (latest)
- BookCorpusOpen

Output structure:
    eign_phaseB_data/
        wiki_en/
            wiki_000001.txt
            wiki_000002.txt
            ...
        books/
            book_000001.txt
            book_000002.txt
            ...

Usage:
    # Download all data
    python scripts/prepare_phaseB_data.py

    # Limit for testing (1000 docs per source)
    python scripts/prepare_phaseB_data.py --max-docs 1000

    # Custom output directory
    python scripts/prepare_phaseB_data.py --output-dir /path/to/data

Requirements:
    pip install datasets tqdm
"""
from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


def clean_text(text: str) -> str:
    """
    Minimal safe cleaning:
    - Strip leading/trailing whitespace
    - Remove excessive blank lines (keep paragraph structure)
    - Keep all semantic content intact
    """
    if not text:
        return ""

    # Strip outer whitespace
    text = text.strip()

    # Reduce multiple consecutive newlines to max 2 (preserve paragraphs)
    lines = []
    prev_empty = False
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            if not prev_empty:
                lines.append("")
                prev_empty = True
        else:
            lines.append(stripped)
            prev_empty = False

    return "\n".join(lines)


def download_wikipedia(
    output_dir: Path, max_docs: int | None = None, language: str = "en"
) -> tuple[int, int]:
    """
    Download Wikipedia English plaintext articles.

    Args:
        output_dir: Directory to save wiki_en/ subfolder
        max_docs: Maximum number of documents (None = all)
        language: Wikipedia language code

    Returns:
        (num_docs, total_chars)
    """
    wiki_dir = output_dir / "wiki_en"
    wiki_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("DOWNLOADING WIKIPEDIA ENGLISH")
    print("=" * 70)
    print(f"Output: {wiki_dir}")
    print(f"Language: {language}")
    print(f"Max docs: {max_docs if max_docs else 'unlimited'}")
    print()

    # Always use streaming mode to avoid downloading entire dataset
    print("[INFO] Using streaming mode (memory and disk efficient)")
    dataset = load_dataset(
        "wikimedia/wikipedia", f"20231101.{language}", split="train", streaming=True
    )

    num_docs = 0
    total_chars = 0

    # Progress bar
    pbar = tqdm(
        total=max_docs if max_docs else None,
        desc="Downloading Wikipedia",
        unit="doc",
    )

    for idx, example in enumerate(dataset):
        if max_docs and idx >= max_docs:
            break

        # Extract text field
        text = example.get("text", "")
        if not text:
            continue

        # Clean minimally
        text = clean_text(text)
        if not text:
            continue

        # Save to file
        file_path = wiki_dir / f"wiki_{idx+1:06d}.txt"
        file_path.write_text(text, encoding="utf-8")

        num_docs += 1
        total_chars += len(text)
        pbar.update(1)

    pbar.close()

    print(f"\n[OK] Wikipedia: {num_docs:,} documents saved")
    print(f"[OK] Total size: {total_chars / 1e6:.1f}M characters")

    return num_docs, total_chars


def download_bookcorpus(
    output_dir: Path, max_docs: int | None = None
) -> tuple[int, int]:
    """
    Download PG-19 books dataset (Project Gutenberg).

    Args:
        output_dir: Directory to save books/ subfolder
        max_docs: Maximum number of books (None = all)

    Returns:
        (num_books, total_chars)
    """
    books_dir = output_dir / "books"
    books_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("DOWNLOADING BOOKS (PG-19)")
    print("=" * 70)
    print(f"Output: {books_dir}")
    print(f"Max books: {max_docs if max_docs else 'unlimited'}")
    print()

    # Always use streaming mode to avoid downloading entire dataset
    print("[INFO] Using streaming mode (memory and disk efficient)")
    dataset = load_dataset("emozilla/pg19", split="train", streaming=True)

    num_books = 0
    total_chars = 0

    # Progress bar
    pbar = tqdm(
        total=max_docs if max_docs else None,
        desc="Downloading Books",
        unit="book",
    )

    for idx, example in enumerate(dataset):
        if max_docs and idx >= max_docs:
            break

        # Extract text field
        text = example.get("text", "")
        if not text:
            continue

        # Clean minimally
        text = clean_text(text)
        if not text:
            continue

        # Save to file (each book is one document)
        file_path = books_dir / f"book_{idx+1:06d}.txt"
        file_path.write_text(text, encoding="utf-8")

        num_books += 1
        total_chars += len(text)
        pbar.update(1)

    pbar.close()

    print(f"\n[OK] Books: {num_books:,} books saved")
    print(f"[OK] Total size: {total_chars / 1e6:.1f}M characters")

    return num_books, total_chars


def print_final_stats(
    output_dir: Path, wiki_stats: tuple[int, int], book_stats: tuple[int, int]
) -> None:
    """Print final statistics."""
    wiki_docs, wiki_chars = wiki_stats
    book_docs, book_chars = book_stats

    total_docs = wiki_docs + book_docs
    total_chars = wiki_chars + book_chars
    total_bytes = 0

    # Calculate actual disk size
    for file in output_dir.rglob("*.txt"):
        total_bytes += file.stat().st_size

    print("\n" + "=" * 70)
    print("FINAL STATISTICS")
    print("=" * 70)
    print(f"[STATS] Wiki docs: {wiki_docs:,}")
    print(f"[STATS] Book docs: {book_docs:,}")
    print(f"[STATS] Total docs: {total_docs:,}")
    print()
    print(f"[STATS] Wiki chars: {wiki_chars / 1e6:.1f}M")
    print(f"[STATS] Book chars: {book_chars / 1e6:.1f}M")
    print(f"[STATS] Total chars: {total_chars / 1e6:.1f}M")
    print()
    print(f"[STATS] Disk size: {total_bytes / 1e9:.2f} GB")
    print(f"[STATS] Avg chars/doc: {total_chars / max(total_docs, 1):,.0f}")
    print()
    print(f"[STATS] Output directory: {output_dir.resolve()}")
    print("=" * 70)
    print()
    print("[INFO] Dataset ready for Kaggle upload!")
    print("[INFO] Next steps:")
    print(f"[INFO]   1. cd {output_dir}")
    print("[INFO]   2. Create Kaggle dataset (Private)")
    print("[INFO]   3. Use in training with --train-data-dir")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare Phase B pretraining data (Wikipedia + BookCorpus)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="eign_phaseB_data",
        help="Output directory for dataset (default: eign_phaseB_data)",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Max documents per source (None = unlimited, for testing use 1000)",
    )
    parser.add_argument(
        "--skip-wiki",
        action="store_true",
        help="Skip Wikipedia download",
    )
    parser.add_argument(
        "--skip-books",
        action="store_true",
        help="Skip BookCorpus download",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("EIGN PHASE B DATA PREPARATION")
    print("=" * 70)
    print(f"Output directory: {output_dir.resolve()}")
    print(f"Max docs per source: {args.max_docs if args.max_docs else 'unlimited'}")
    print()

    # Download Wikipedia
    wiki_stats = (0, 0)
    if not args.skip_wiki:
        wiki_stats = download_wikipedia(output_dir, args.max_docs)
    else:
        print("\n[SKIP] Wikipedia download skipped")

    # Download BookCorpus
    book_stats = (0, 0)
    if not args.skip_books:
        book_stats = download_bookcorpus(output_dir, args.max_docs)
    else:
        print("\n[SKIP] BookCorpus download skipped")

    # Print final statistics
    print_final_stats(output_dir, wiki_stats, book_stats)


if __name__ == "__main__":
    main()
