import os
import json
import math
import logging
import random
from dataclasses import dataclass
from typing import List, Dict, Optional

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# =========================================================
# 1. Config
# =========================================================

@dataclass
class SFTConfig:
    model_path: str
    train_file: str
    output_dir: str = "./output"

    max_length: int = 2048
    batch_size: int = 2
    grad_accum_steps: int = 8

    lr: float = 2e-5
    epochs: int = 3
    warmup_ratio: float = 0.05

    fp16: bool = True
    bf16: bool = False

    log_steps: int = 10
    save_steps: int = 500

    num_workers: int = 4
    seed: int = 42


# =========================================================
# 2. Utils
# =========================================================

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def clean_text(text: str) -> str:
    return text.strip()


# =========================================================
# 3. Tokenization Template
# =========================================================

def build_prompt(prompt: str, response: str) -> str:
    return f"""### Instruction:
{prompt}

### Response:
{response}"""


# =========================================================
# 4. Dataset (SFT + Mask loss)
# =========================================================

class SFTDataset(Dataset):
    def __init__(self, file_path: str, tokenizer, max_length: int):
        self.data = load_json(file_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def _encode(self, text: str):
        return self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )

    def __getitem__(self, idx):
        item = self.data[idx]

        prompt = clean_text(item["prompt"])
        response = clean_text(item["response"])

        full_text = build_prompt(prompt, response)

        enc = self._encode(full_text)

        input_ids = torch.tensor(enc["input_ids"])
        attention_mask = torch.tensor(enc["attention_mask"])

        labels = input_ids.clone()

        # mask prompt loss
        prompt_ids = self.tokenizer(
            build_prompt(prompt, ""),
            add_special_tokens=False
        )["input_ids"]

        prompt_len = min(len(prompt_ids), len(labels))

        labels[:prompt_len] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


# =========================================================
# 5. Collator
# =========================================================

class DataCollator:
    def __call__(self, batch):
        return {
            "input_ids": torch.stack([x["input_ids"] for x in batch]),
            "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
            "labels": torch.stack([x["labels"] for x in batch]),
        }


# =========================================================
# 6. Model Loader
# =========================================================

def load_model(model_path: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    return model


# =========================================================
# 7. Training Engine
# =========================================================

class SFTTrainer:
    def __init__(self, config: SFTConfig):
        self.config = config
        set_seed(config.seed)

        self.tokenizer = AutoTokenizer.from_pretrained(config.model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = load_model(config.model_path)

        self.dataset = SFTDataset(
            config.train_file,
            self.tokenizer,
            config.max_length
        )

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            collate_fn=DataCollator()
        )

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.lr
        )

        total_steps = len(self.dataloader) * config.epochs
        warmup_steps = int(total_steps * config.warmup_ratio)

        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------------------------------------

    def compute_loss(self, outputs, labels):
        logits = outputs.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)

        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )

        return loss

    # -----------------------------------------------------

    def train_step(self, batch):
        batch = {k: v.to(self.device) for k, v in batch.items()}

        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        )

        loss = self.compute_loss(outputs, batch["labels"])
        loss.backward()

        return loss.item()

    # -----------------------------------------------------

    def save(self, step):
        path = os.path.join(self.config.output_dir, f"step_{step}")
        os.makedirs(path, exist_ok=True)

        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    # -----------------------------------------------------

    def train(self):
        self.model.train()

        global_step = 0
        accum_loss = 0

        os.makedirs(self.config.output_dir, exist_ok=True)

        for epoch in range(self.config.epochs):
            logging.info(f"Epoch {epoch} start")

            for step, batch in enumerate(self.dataloader):

                loss = self.train_step(batch)
                accum_loss += loss

                if (step + 1) % self.config.grad_accum_steps == 0:
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    global_step += 1

                    if global_step % self.config.log_steps == 0:
                        logging.info(
                            f"step={global_step}, loss={accum_loss / self.config.log_steps:.4f}"
                        )
                        accum_loss = 0

                    if global_step % self.config.save_steps == 0:
                        self.save(global_step)

        self.save("final")


# =========================================================
# 8. Evaluation (¼òµ¥ perplexity)
# =========================================================

@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    losses = []

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        )

        logits = outputs.logits

        shift_logits = logits[..., :-1, :]
        shift_labels = batch["labels"][..., 1:]

        loss = torch.nn.functional.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            ignore_index=-100
        )

        losses.append(loss.item())

    model.train()
    return math.exp(sum(losses) / len(losses))


# =========================================================
# 9. Main
# =========================================================

def main():
    config = SFTConfig(
        model_path="/path/to/model",
        train_file="/path/to/data.json",
        output_dir="./sft_output"
    )

    trainer = SFTTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()