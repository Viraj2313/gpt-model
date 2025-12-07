import os
import torch
from tokenizers import Tokenizer
from huggingface_hub import HfApi, login, hf_hub_download
from kaggle_secrets import UserSecretsClient

class Trainer:
    def __init__(
        self,
        tokenizer_path,
        train_data_path,
        val_data_path,
        ckpt_repo,
        local_ckpt,
        local_log,
        max_iters,
        model_class,
        batch_size=16,
        block_size=512,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.train_data = torch.load(train_data_path, map_location="cpu")
        self.val_data = torch.load(val_data_path, map_location="cpu")
        self.batch_size = batch_size
        self.block_size = block_size
        self.vocab_size = self.tokenizer.get_vocab_size()
        self.api = HfApi()

        user_secrets = UserSecretsClient()
        token = user_secrets.get_secret("HF_TOKEN")
        login(token=token)

        self.ckpt_repo = ckpt_repo
        self.local_ckpt = local_ckpt
        self.local_log = local_log
        self.max_iters = max_iters

        if os.path.exists(self.local_ckpt):
            checkpoint_path = self.local_ckpt
        else:
            checkpoint_path = hf_hub_download(
                repo_id=self.ckpt_repo,
                repo_type="model",
                filename="checkpoint.pth"
            )
            torch.save(torch.load(checkpoint_path, map_location=self.device), self.local_ckpt)

        if os.path.exists(self.local_log):
            log_path = self.local_log
        else:
            log_path = hf_hub_download(
                repo_id=self.ckpt_repo,
                repo_type="model",
                filename="train_log.txt"
            )
            with open(log_path, "r") as src, open(self.local_log, "w") as dst:
                dst.write(src.read())

        self.model = model_class(
            vocab_size=self.vocab_size,
            n_embd=768,
            block_size=self.block_size,
            n_layers=12,
            n_heads=12,
            dropout=0.2
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=6e-4, weight_decay=0.1)

        checkpoint = torch.load(self.local_ckpt, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.start_step = checkpoint["step"]

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.max_iters,
            eta_min=6e-5,
            last_epoch=self.start_step - 1
        )

        self.logfile = open(self.local_log, "a")

    def get_batch(self, split):
        d = self.train_data if split == "train" else self.val_data
        ix = torch.randint(0, len(d) - self.block_size, (self.batch_size,))
        x = torch.stack([d[i:i+self.block_size] for i in ix])
        y = torch.stack([d[i+1:i+self.block_size+1] for i in ix])
        return x.to(self.device), y.to(self.device)

    def upload(self, path, filename):
        self.api.upload_file(
            path_or_fileobj=path,
            path_in_repo=filename,
            repo_id=self.ckpt_repo,
            repo_type="model"
        )

    def train(self):
        for step in range(self.start_step, self.max_iters + 1):
            self.model.train()
            xb, yb = self.get_batch("train")
            logits, loss = self.model(xb, yb)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()

            if step % 200 == 0:
                self.model.eval()
                with torch.no_grad():
                    vx, vy = self.get_batch("val")
                    _, v_loss = self.model(vx, vy)

                msg = f"Step {step}: Train Loss {loss.item():.4f}, Val Loss {v_loss.item():.4f}"
                print(msg)
                self.logfile.write(msg + "\n")
                self.logfile.flush()

                torch.save(
                    {"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(), "step": step},
                    self.local_ckpt
                )

                self.upload(self.local_ckpt, "checkpoint.pth")
                self.upload(self.local_log, "train_log.txt")
                print("Checkpoint saved.")


def run_training(model_class):
    trainer = Trainer(
        tokenizer_path="/kaggle/working/my_tokenizer.json",
        train_data_path="/kaggle/working/train_data.pt",
        val_data_path="/kaggle/working/val_data.pt",
        ckpt_repo="viraj231/gpt-100m",
        local_ckpt="/kaggle/working/checkpoint.pth",
        local_log="/kaggle/working/train_log.txt",
        max_iters=100000,
        model_class=model_class
    )
    trainer.train()


if __name__ == "__main__":
    from model import GPTLM
    run_training(GPTLM)
