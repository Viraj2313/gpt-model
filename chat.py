"""
chat interface.
"""
from src.infer.generate import load_model_for_inference

print("Loading model...")
generator = load_model_for_inference(
    checkpoint_path="checkpoints/checkpoint.pth",
    tokenizer_path="bpe_tokenizer.json"
)

print("\n" + "=" * 60)
print("Chat started. Type 'exit', 'quit', or 'stop' to end.")
print("=" * 60 + "\n")

history = ""

while True:
    user_input = input("User: ").strip()
    
    if user_input.lower() in ["exit", "quit", "stop"]:
        print("Ending chat.")
        break
    
    if not user_input:
        continue
    
    reply, history = generator.chat_turn(
        history,
        user_input,
        max_new_tokens=150,
        temperature=0.8,
        top_k=50
    )
    
    print(f"Assistant: {reply}\n")