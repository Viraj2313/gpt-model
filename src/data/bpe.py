import os
import torch
import numpy as np
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, normalizers
from huggingface_hub import HfApi, HfFolder

def train_tokenizer(input_file, temp_dir, repo):
    api = HfApi()
    token = HfFolder.get_token()

    tokenizer = Tokenizer(models.BPE())
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=16000,
        min_frequency=2,
        special_tokens=["<|endoftext|>", "<|pad|>", "<|user|>", "<|assistant|>", "<|system|>"]
    )

    def corpus():
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                yield line

    tokenizer.train_from_iterator(corpus(), trainer)

    tokenizer_path = os.path.join(temp_dir, "my_tokenizer.json")
    tokenizer.save(tokenizer_path)

    api.upload_file(
        path_or_fileobj=tokenizer_path,
        path_in_repo="my_tokenizer.json",
        repo_id=repo,
        repo_type="dataset",
        token=token
    )

    return tokenizer


def create_shards(input_file, tokenizer, temp_dir, repo, shard_size=250*1024*1024, batch_size=50000):
    api = HfApi()
    token = HfFolder.get_token()

    shard_index = 0
    current = 0
    shard_path = os.path.join(temp_dir, f"clean_{shard_index:05d}.bin")
    shard_file = open(shard_path, "ab")
    batch = []

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            batch.append(line)
            if len(batch) >= batch_size:
                enc = tokenizer.encode_batch(batch)
                ids = np.concatenate([np.array(e.ids, dtype=np.uint16) for e in enc])
                b = ids.tobytes()

                if current + len(b) > shard_size:
                    shard_file.close()
                    api.upload_file(
                        path_or_fileobj=shard_path,
                        path_in_repo=f"shards/{os.path.basename(shard_path)}",
                        repo_id=repo,
                        repo_type="dataset",
                        token=token
                    )
                    os.remove(shard_path)

                    shard_index += 1
                    current = 0
                    shard_path = os.path.join(temp_dir, f"clean_{shard_index:05d}.bin")
                    shard_file = open(shard_path, "ab")

                shard_file.write(b)
                current += len(b)
                batch = []

    if batch:
        enc = tokenizer.encode_batch(batch)
        ids = np.concatenate([np.array(e.ids, dtype=np.uint16) for e in enc])
        shard_file.write(ids.tobytes())

    shard_file.close()

    api.upload_file(
        path_or_fileobj=shard_path,
        path_in_repo=f"shards/{os.path.basename(shard_path)}",
        repo_id=repo,
        repo_type="dataset",
        token=token
    )

    os.remove(shard_path)


def merge_shards(temp_dir, repo):
    api = HfApi()

    shards = []
    for f in api.list_files_info(repo, repo_type="dataset"):
        if f.rfilename.startswith("shards/"):
            path = api.hf_hub_download(
                repo_id=repo,
                filename=f.rfilename,
                repo_type="dataset",
                local_dir=temp_dir
            )
            shards.append(path)

    tokens = []
    for s in sorted(shards):
        raw = np.memmap(s, dtype=np.uint16, mode="r")
        tokens.append(torch.from_numpy(raw.astype(np.int64)))

    return torch.cat(tokens, dim=0)


def create_train_val(data, temp_dir, repo):
    api = HfApi()
    token = HfFolder.get_token()

    n = int(0.9 * len(data))
    train = data[:n].clone().cpu()
    val = data[n:].clone().cpu()

    train_path = os.path.join(temp_dir, "train_data.pt")
    val_path = os.path.join(temp_dir, "val_data.pt")

    torch.save(train, train_path, _use_new_zipfile_serialization=False)
    torch.save(val, val_path, _use_new_zipfile_serialization=False)

    api.upload_file(
        path_or_fileobj=train_path,
        path_in_repo="train_data.pt",
        repo_id=repo,
        repo_type="dataset",
        token=token
    )

    api.upload_file(
        path_or_fileobj=val_path,
        path_in_repo="val_data.pt",
        repo_id=repo,
        repo_type="dataset",
        token=token
    )


def run_pipeline():
    input_file = "phase1.txt"
    temp_dir = "/kaggle/temp/llm_all"
    repo = "viraj231/gpt"

    os.makedirs(temp_dir, exist_ok=True)

    tokenizer = train_tokenizer(input_file, temp_dir, repo)
    create_shards(input_file, tokenizer, temp_dir, repo)
    data = merge_shards(temp_dir, repo)
    create_train_val(data, temp_dir, repo)
    print("done")


if __name__ == "__main__":
    run_pipeline()
