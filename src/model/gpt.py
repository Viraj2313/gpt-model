import torch
import torch.nn as nn
from torch.nn import functional as F

class Head(nn.Module):
    """ One head of self-attention """

    def __init__(self, head_size, n_embd, block_size):
        super().__init__()

        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        # causal mask: to prevent attention to future tokes
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        B, T, C = x.shape

        # compute q, k, v projections
        k = self.key(x)     # (B, T, head_size)
        q = self.query(x)   # (B, T, head_size)

        # compute attention scores
        # (B, T, head_size) @ (B, head_size, T) = (B, T, T)
        wei = q @ k.transpose(-2, -1)

        # scale
        wei = wei / (k.shape[-1] ** 0.5)

        # apply mask
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))

        # softmax
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        # weighted sum over values
        v = self.value(x)  # (B, T, head_size)
        out = wei @ v      # (B, T, head_size)

        return out


class MultiHeadAttention(nn.Module):
    """ Multiple attention heads in parallel """

    def __init__(self, num_heads, head_size, n_embd, block_size):
        super().__init__()
        self.heads = nn.ModuleList([
            Head(head_size, n_embd, block_size)
            for _ in range(num_heads)
        ])
        self.proj = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # concatenate all head outputs on the feature dimension
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """ feed-forward network """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),   # activation
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication + computation """

    def __init__(self, n_embd, num_heads, block_size):
        super().__init__()

        head_size = n_embd // num_heads

        self.sa = MultiHeadAttention(num_heads, head_size, n_embd, block_size)
        self.ffn = FeedForward(n_embd)

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # first sub-layer: self-attention
        x = x + self.sa(self.ln1(x))

        # second sub-layer: feed-forward
        x = x + self.ffn(self.ln2(x))

        return x


class GPTLM(nn.Module):
    def __init__(self, vocab_size, n_embd, block_size, n_layers, n_heads, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        self.n_embd = n_embd

        # embeddings
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        # transformer blocks
        self.blocks = nn.ModuleList([
            Block(n_embd=n_embd, num_heads=n_heads, block_size=block_size)
            for _ in range(n_layers)
        ])

        # final norm and linear head
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        self.lm_head.weight = self.token_embedding_table.weight

        self.dropout = nn.Dropout(dropout)

    def forward(self, idx, targets=None):
        """
        idx: (B, T) token indices
        targets: (B, T) token indices (next tokens), optional

        Returns:
          - if targets is None: logits (B, T, vocab_size)
          - else: (logits_flat, loss)
        """
        B, T = idx.shape
        assert T <= self.block_size, "input length T must be <= block_size"

        # token + position embeddings
        tok_emb = self.token_embedding_table(idx)          # (B, T, n_embd)
        pos = torch.arange(T, device=idx.device)           # (T,)
        pos_emb = self.position_embedding_table(pos)       # (T, n_embd)
        x = tok_emb + pos_emb                              # broadcasting -> (B, T, n_embd)
        x = self.dropout(x)

        # transformer blocks
        for block in self.blocks:
            x = block(x)

        # final norm, linear head to vocab
        x = self.ln_f(x)                                   # (B, T, n_embd)
        logits = self.lm_head(x)                           # (B, T, vocab_size)

        if targets is None:
            return logits

        # compute loss (flatten B*T)
        B, T, C = logits.shape
        logits_flat = logits.view(B * T, C)
        targets_flat = targets.view(B * T)
        loss = F.cross_entropy(logits_flat, targets_flat)
        return logits_flat, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, stop_token_ids=None):
        for _ in range(max_new_tokens):
            if idx.size(1) > self.block_size:
                idx_cond = idx[:, -self.block_size:]
            else:
                idx_cond = idx
            
            logits = self(idx_cond)
            logits = logits[:, -1, :]
            
            if temperature != 1.0:
                logits = logits / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                min_topk = v[:, -1].unsqueeze(-1)
                logits = torch.where(logits < min_topk, torch.full_like(logits, -1e10), logits)
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Check for stop tokens
            if stop_token_ids and next_token.item() in stop_token_ids:
                break
                
            idx = torch.cat((idx, next_token), dim=1)
        
        return idx