"""
training script.
"""
import torch
from src.training.load import setup_training, load_checkpoint, save_checkpoint

print("Setting up training...")
model, optimizer, get_batch, device, vocab_size = setup_training(
    tokenizer_path="bpe_tokenizer.json",
    corpus_path="datacorpus.txt",
    batch_size=64,
    block_size=256,
    n_embd=768,
    n_layers=8,
    n_heads=8,
    dropout=0.1,
    learning_rate=3e-4
)

start_step = load_checkpoint(
    model,
    optimizer,
    checkpoint_path="checkpoints/checkpoint.pth",
    device=device
)

max_steps = 55000
eval_interval = 200
save_interval = 200

print(f"\nStarting training from step {start_step} to {max_steps}")
print(f"Eval/Save every {eval_interval} steps")
print("-" * 60)

model.train()

for step in range(start_step, max_steps):
    xb, yb = get_batch("train")
    
    logits, loss = model(xb, yb)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if step % eval_interval == 0:
        print(f"Step {step}: loss = {loss.item():.4f}")
        
        save_checkpoint(
            model,
            optimizer,
            step,
            loss.item(),
            checkpoint_path="checkpoints/checkpoint.pth"
        )

save_checkpoint(
    model,
    optimizer,
    max_steps,
    loss.item(),
    checkpoint_path="checkpoints/checkpoint.pth"
)

print("\nTraining complete!")