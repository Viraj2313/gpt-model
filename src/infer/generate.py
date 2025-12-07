"""
Text generation and inference utilities.
"""
import torch
from tokenizers import Tokenizer
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.model.gpt import GPTLM


class Generator:
    def __init__(
        self,
        model: GPTLM,
        tokenizer: Tokenizer,
        device: str = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        self.model.eval()
    
    def encode(self, text: str) -> torch.Tensor:
        """Encode text to token IDs."""
        ids = self.tokenizer.encode(text).ids
        return torch.tensor([ids], dtype=torch.long).to(self.device)
    
    def decode(self, ids: list) -> str:
        """Decode token IDs to text."""
        return self.tokenizer.decode(ids)
    
    def generate(
        self,
        prompt: str = "",
        max_new_tokens: int = 150,
        temperature: float = 0.8,
        top_k: int = 50
    ) -> str:
     
        if not prompt:
            bos_id = self.tokenizer.token_to_id("<bos>")
            if bos_id is None:
                bos_id = 0
            context = torch.tensor([[bos_id]], dtype=torch.long).to(self.device)
        else:
            context = self.encode(prompt)
        
        with torch.no_grad():
            output = self.model.generate(
                context,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k
            )
        
        generated_text = self.decode(output[0].tolist())
        
        return generated_text
    
    def chat_turn(
        self,
        history: str,
        user_message: str,
        max_new_tokens: int = 150,
        temperature: float = 0.8,
        top_k: int = 50
    ) -> tuple:
   
        history += f"User: {user_message}\nAssistant:"
        
        full_text = self.generate(
            prompt=history,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k
        )
        
        if "Assistant:" in full_text:
            assistant_reply = full_text.split("Assistant:", 1)[1].strip()
        else:
            assistant_reply = full_text
        
        if "User:" in assistant_reply:
            assistant_reply = assistant_reply.split("User:")[0].strip()
        
        history += " " + assistant_reply + "\n"
        
        return assistant_reply, history


def load_model_for_inference(
    checkpoint_path: str = "checkpoints/checkpoint.pth",
    tokenizer_path: str = "bpe_tokenizer.json",
    device: str = None
) -> Generator:
  
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    device_obj = torch.device(device)
    
    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    
    n_embd = 768
    block_size = 256
    n_layers = 8
    n_heads = 8
    dropout = 0.1
    
    print("Creating model...")
    model = GPTLM(
        vocab_size=vocab_size,
        n_embd=n_embd,
        block_size=block_size,
        n_layers=n_layers,
        n_heads=n_heads,
        dropout=dropout
    )
    
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device_obj)
    model.load_state_dict(checkpoint["model"])
    
    step = checkpoint.get('step', 'unknown')
    print(f"Model loaded (step {step})")
    
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model parameters: {params:.2f}M")
    
    generator = Generator(model, tokenizer, device)
    
    return generator


def interactive_chat():
    print("Loading model...")
    generator = load_model_for_inference()
    
    print("\n" + "=" * 60)
    print("GPT Chat Interface")
    print("=" * 60)
    print("Type 'exit', 'quit', or 'stop' to end the conversation.\n")
    
    history = ""
    
    while True:
        user_input = input("User: ").strip()
        
        if user_input.lower() in ["exit", "quit", "stop"]:
            print("Ending chat. Goodbye!")
            break
        
        if not user_input:
            continue
        
        reply, history = generator.chat_turn(history, user_input)
        print(f"Assistant: {reply}\n")


def simple_generation():
    print("Loading model...")
    generator = load_model_for_inference()
    
    print("\nGenerating text...\n")
    
    text = generator.generate(
        prompt="",
        max_new_tokens=300,
        temperature=0.8,
        top_k=50
    )
    
    print(text)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--generate":
        simple_generation()
    else:
        interactive_chat()