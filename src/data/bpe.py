"""
BPE Tokenizer training module.
"""
import torch
from tokenizers import Tokenizer, models, trainers, pre_tokenizers


def train_bpe_tokenizer(
    corpus_path: str = "datacorpus.txt",
    output_path: str = "bpe_tokenizer.json",
    vocab_size: int = 8000,
    min_frequency: int = 2
):
    """Train and save BPE tokenizer."""
    print(f"Loading corpus from {corpus_path}...")
    with open(corpus_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    print("Training BPE tokenizer...")
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"]
    )
    
    tokenizer.train_from_iterator([text], trainer)
    tokenizer.save(output_path)
    
    print(f"Tokenizer saved to {output_path}")
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
    
    return tokenizer


if __name__ == "__main__":
    train_bpe_tokenizer()