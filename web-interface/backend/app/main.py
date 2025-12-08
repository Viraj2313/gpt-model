from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from app.model import GPTLM

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = "cpu"
tokenizer = Tokenizer.from_file("my_tokenizer.json")
vocab_size = tokenizer.get_vocab_size()

model = GPTLM(
    vocab_size=vocab_size,
    n_embd=768,
    block_size=512,
    n_layers=12,
    n_heads=12,
    dropout=0.2
).to(device)

state = torch.load("tech_model_final.pth", map_location=device, weights_only=True)
model.load_state_dict(state)
model.eval()

class Prompt(BaseModel):
    prompt: str

def stream_generate(user_input):
    prompt = f"<|user|>\n{user_input}\n<|assistant|>\n"
    
    ids = tokenizer.encode(prompt).ids
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    prompt_len = idx.shape[1]
    
    with torch.no_grad():
        for _ in range(200):
            logits = model(idx[:, -512:])
            logits = logits[:, -1, :] / 0.6
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, 1)
            
            token = tokenizer.decode([idx_next.item()])
            
            if token.strip() == "" or token.startswith("\n\n"):
                break
            if "<|endoftext|>" in token or "<|user|>" in token:
                break
            
            if idx.shape[1] >= prompt_len:
                yield token
            
            idx = torch.cat((idx, idx_next), dim=1)

@app.post("/generate-stream")
def generate_stream(req: Prompt):
    return StreamingResponse(stream_generate(req.prompt), media_type="text/plain")
