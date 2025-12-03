"""
Data preparation module for GPT training.
Handles dataset loading and preprocessing.
"""
import re
import torch
from datasets import load_dataset
from typing import Tuple


def clean(s: str) -> str:
    """Clean text by removing tabs, extra spaces, and URLs."""
    s = s.replace("\t", " ").replace("  ", " ")
    s = re.sub(r"http\S+", "", s)
    return s.strip()


def prepare_dailydialog() -> list:
    """
    Load and process DailyDialog dataset into ChatGPT-style format.
    
    Returns:
        List of formatted chat lines
    """
    print("Loading DailyDialog...")
    dd = load_dataset("daily_dialog")["train"]
    
    chat_lines = []
    
    # Convert DailyDialog into ChatGPT style
    for dialog in dd["dialog"]:
        turns = [t.strip() for t in dialog if t.strip()]
        speaker = "User"
        for t in turns:
            chat_lines.append(f"{speaker}: {clean(t)}")
            speaker = "Assistant" if speaker == "User" else "User"
        chat_lines.append("")
    
    print("DailyDialog processed.")
    return chat_lines


def prepare_wikitext(max_chars: int = 200_000) -> list:
    """
    Load and process WikiText-2 dataset.
    
    Args:
        max_chars: Maximum number of characters to use
        
    Returns:
        List of formatted sentences
    """
    print("Loading WikiText-2 (smaller, safe)...")
    wiki = load_dataset("wikitext", "wikitext-2-raw-v1")["train"]["text"]
    wiki_raw = clean("\n".join(wiki))
    wiki_raw = wiki_raw[:max_chars]
    
    wiki_sents = re.split(r'(?<=[.!?])\s+', wiki_raw)
    chat_lines = []
    
    for s in wiki_sents:
        s = s.strip()
        if len(s) > 5:
            chat_lines.append(f"Assistant: {s}")
    chat_lines.append("")
    
    return chat_lines


def create_corpus(output_path: str = "datacorpus.txt") -> Tuple[str, int]:
    """
    Create the full training corpus by combining datasets.
    
    Args:
        output_path: Path to save the corpus file
        
    Returns:
        Tuple of (corpus text, size in bytes)
    """
    dailydialog_lines = prepare_dailydialog()
    wikitext_lines = prepare_wikitext()
    
    all_lines = dailydialog_lines + wikitext_lines
    final_text = "\n".join(all_lines)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final_text)
    
    size = len(final_text)
    print(f"Saved {output_path} successfully!")
    print(f"Size (MB): {size / 1e6:.2f}")
    
    return final_text, size


def load_and_split_data(
    corpus_path: str, 
    tokenizer,
    train_split: float = 0.9
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load corpus, tokenize, and split into train/val sets.
    
    Args:
        corpus_path: Path to the corpus text file
        tokenizer: Trained tokenizer instance
        train_split: Fraction of data to use for training
        
    Returns:
        Tuple of (train_data, val_data) as torch tensors
    """
    with open(corpus_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    ids = tokenizer.encode(text).ids
    data = torch.tensor(ids, dtype=torch.long)
    
    n = int(train_split * len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    print(f"Train tokens: {len(train_data):,}")
    print(f"Val tokens: {len(val_data):,}")
    
    return train_data, val_data


if __name__ == "__main__":
    create_corpus()