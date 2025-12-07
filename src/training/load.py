import torch
from tokenizers import Tokenizer

def load_environment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.from_file("/kaggle/working/my_tokenizer.json")
    train_data = torch.load("/kaggle/working/train_data.pt", map_location="cpu")
    val_data = torch.load("/kaggle/working/val_data.pt", map_location="cpu")
    batch_size = 16
    block_size = 512
    vocab_size = tokenizer.get_vocab_size()
    return device, tokenizer, train_data, val_data, batch_size, block_size, vocab_size

def get_batch(split, train_data, val_data, batch_size, block_size, device):
    d = train_data if split == "train" else val_data
    ix = torch.randint(0, len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i+block_size] for i in ix])
    y = torch.stack([d[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

def load_model(model_class):
    device, tokenizer, train_data, val_data, batch_size, block_size, vocab_size = load_environment()
    model = model_class(
        vocab_size=vocab_size,
        n_embd=768,
        block_size=block_size,
        n_layers=12,
        n_heads=12,
        dropout=0.2
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=0.1)
    return model, optimizer, get_batch, device, train_data, val_data, batch_size, block_size, vocab_size

if __name__ == "__main__":
    from model import GPTLM
    model, optimizer, get_batch, device, train_data, val_data, batch_size, block_size, vocab_size = load_model(GPTLM)
    print("Device:", device)
    print("Model parameters:", sum(p.numel() for p in model.parameters()) / 1e6, "M")
