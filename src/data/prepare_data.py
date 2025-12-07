import random
from datasets import load_dataset, Dataset, concatenate_datasets
from tqdm import tqdm

identity_samples_phase1 = [
    "User: Who are you?\nAssistant: I am an AI Assistant.",
    "User: What is your name?\nAssistant: I am an AI Assistant designed to help with science and technology.",
    "User: Are you a human?\nAssistant: No, I am an AI Assistant.",
    "User: Hi\nAssistant: Hello! I am an AI Assistant. How can I help you?",
    "User: What are you?\nAssistant: I am an AI Assistant.",
    "User: Introduce yourself.\nAssistant: I am an AI Assistant trained to explain complex topics."
]

identity_samples_phase2 = [
    {"messages": [{"role": "user", "content": "Who are you?"}, {"role": "assistant", "content": "I am an AI Assistant."}]},
    {"messages": [{"role": "user", "content": "What is your name?"}, {"role": "assistant", "content": "I am an AI Assistant designed to help with science."}]},
    {"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello! I am an AI Assistant."}]},
    {"messages": [{"role": "user", "content": "Are you human?"}, {"role": "assistant", "content": "No, I am an artificial intelligence."}]}
] * 100


def generate_phase1(output_path="phase1.txt", limit=1_500_000):
    ds = load_dataset("HuggingFaceTB/cosmopedia", "web_samples_v2", split="train", streaming=True)
    it = iter(ds)

    with open(output_path, "w", encoding="utf-8") as f:
        for i in range(limit):
            try:
                if i > 0 and i % 100 == 0:
                    text = random.choice(identity_samples_phase1)
                else:
                    text = next(it)["text"]
                f.write(text + " <|endoftext|>\n")
            except StopIteration:
                break

    return output_path


def generate_phase2(output_path="phase2.txt"):
    ds_identity = Dataset.from_list(identity_samples_phase2)
    ds_chat_stream = load_dataset("HuggingFaceTB/smoltalk", "everyday-conversations", split="train", streaming=True)
    ds_chat = Dataset.from_list(list(ds_chat_stream.take(20000)))

    combined = concatenate_datasets([ds_chat, ds_identity]).shuffle(seed=42)

    with open(output_path, "w", encoding="utf-8") as f:
        for example in tqdm(combined):
            text = ""
            for msg in example["messages"]:
                if msg["role"] == "user":
                    text += f"<|user|>\n{msg['content']}\n"
                elif msg["role"] == "assistant":
                    text += f"<|assistant|>\n{msg['content']}\n"
            text += "<|endoftext|>\n"
            f.write(text)

    return output_path


def run_all():
    generate_phase1("phase1.txt")
    generate_phase2("phase2.txt")
    print("done")


if __name__ == "__main__":
    run_all()
