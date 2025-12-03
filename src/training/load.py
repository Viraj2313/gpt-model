"""
Model and data loading utilities for training.
"""
import torch
from tokenizers import Tokenizer
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.model.gpt import GPTLM


def setup_training(
    tokenizer_path: str = "bpe_tokenizer.json",
    corpus_path: str = "datacorpus.txt",
    batch_size: int = 64,
    block_size: int = 256,
    n_embd: int = 768,
    n_layers: int = 8,
    n_heads: int = 8,
    dropout: float = 0.1,
    learning_rate: float = 3e-4,
    device: str = None
):
    """
    Setup everything needed for training.
    
    Args:
        tokenizer_path: Path to trained tokenizer
        corpus_path: Path to training corpus
        batch_size: Batch size for training
        block_size: Context window size
        n_embd: Embedding dimension
        n_layers: Number of transformer layers
        n_heads: Number of attention heads
        dropout: Dropout rate
        learning_rate: Learning rate for optimizer
        device: Device to use (auto-detect if None)
        
    Returns:
        Tuple of (model, optimizer, get_batch_fn, device, vocab_size)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    
    print("Device:", device)
    
    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    print(f"Vocabulary size: {vocab_size}")
    
    print(f"Loading corpus from {corpus_path}...")
    with open(corpus_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    print("Tokenizing corpus...")
    ids = tokenizer.encode(text).ids
    data = torch.tensor(ids, dtype=torch.long)
    
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    print(f"Train tokens: {len(train_data):,}")
    print(f"Val tokens: {len(val_data):,}")
    
    def get_batch(split):
        """Get a batch of training data."""
        d = train_data if split == "train" else val_data
        ix = torch.randint(0, len(d) - block_size - 1, (batch_size,))
        x = torch.stack([d[i:i+block_size] for i in ix]).to(device)
        y = torch.stack([d[i+1:i+block_size+1] for i in ix]).to(device)
        return x, y
    
    print("Creating model...")
    model = GPTLM(
        vocab_size=vocab_size,
        n_embd=n_embd,
        block_size=block_size,
        n_layers=n_layers,
        n_heads=n_heads,
        dropout=dropout
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model parameters: {params:.2f}M")
    
    return model, optimizer, get_batch, device, vocab_size


def load_checkpoint(
    model,
    optimizer,
    checkpoint_path: str = "checkpoints/checkpoint.pth",
    device: str = None
):
    """
    Load model checkpoint.
    
    Args:
        model: Model to load weights into
        optimizer: Optimizer to load state into
        checkpoint_path: Path to checkpoint file
        device: Device to load on
        
    Returns:
        Step number from checkpoint (0 if no checkpoint found)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        print(f"No checkpoint found at {checkpoint_path}")
        print("Starting training from scratch...")
        return 0
    
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    
    step = checkpoint.get("step", 0)
    loss = checkpoint.get("loss", None)
    
    print(f"Checkpoint loaded at step: {step}")
    if loss is not None:
        print(f"Previous loss: {loss:.4f}")
    
    return step


def save_checkpoint(
    model,
    optimizer,
    step: int,
    loss: float = None,
    checkpoint_path: str = "checkpoints/checkpoint.pth"
):
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer to save
        step: Current training step
        loss: Current loss value (optional)
        checkpoint_path: Path to save checkpoint
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step
    }
    
    if loss is not None:
        checkpoint["loss"] = loss
    
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at step {step}")


if __name__ == "__main__":
    model, optimizer, get_batch, device, vocab_size = setup_training()
    
    print("\nSetup complete!")
    print(f"Device: {device}")
    print(f"Vocab size: {vocab_size}")
    
    print("\nTesting batch loading...")
    x, y = get_batch("train")
    print(f"Batch shape: x={x.shape}, y={y.shape}")
    
    print("\nTesting forward pass...")
    with torch.no_grad():
        logits = model(x)
        print(f"Output shape: {logits.shape}")