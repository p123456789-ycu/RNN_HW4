import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig, TrainerCallback
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from transformers import TrainingArguments, Trainer
from torch.utils.data import Dataset as TorchDataset
from qwen_vl_utils import process_vision_info
import sys

# ── Log setup ────────────────────────────────────────────
class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    def isatty(self):
        return False

logger = Logger("log_train.txt")
sys.stdout = logger
sys.stderr = logger

# ── 1. Quantization Config ───────────────────────────────
model_id = "Qwen/Qwen2-VL-7B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# ── 2. Load Model & Processor ────────────────────────────
# ✅ Qwen2-VL 不需要 trust_remote_code，介面跟 LLaVA 很像
print("Loading Qwen2-VL-7B-Instruct in 4-bit...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_id)
print("Model loaded.")

# ── 3. LoRA Config ───────────────────────────────────────
model = prepare_model_for_kbit_training(model)

# 凍結 Visual Encoder
for name, param in model.named_parameters():
    if "visual" in name:
        param.requires_grad = False

# ✅ Qwen2-VL LLM 部分是標準 attention，直接用 q_proj/v_proj
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# ── 4. Custom Dataset ────────────────────────────────────
class ChartQADataset(TorchDataset):
    def __init__(self, hf_dataset, processor):
        self.dataset   = hf_dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample   = self.dataset[idx]
        image    = sample['image'].convert("RGB")
        question = sample['query']
        answer   = sample['label'][0]

        # ── Qwen2-VL 的 messages 格式
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text",  "text": question}
                ]
            }
        ]

        # ── Prompt-only（用來算 prompt 長度，做 label masking）
        prompt_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, _ = process_vision_info(messages)
        prompt_inputs = self.processor(
            text=[prompt_text],
            images=image_inputs,
            return_tensors="pt",
            padding=False
        )
        prompt_len = prompt_inputs["input_ids"].shape[1]

        # ── Full sequence（含 answer）
        full_messages = messages + [{"role": "assistant", "content": answer}]
        full_text = self.processor.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False
        )
        full_inputs = self.processor(
            text=[full_text],
            images=image_inputs,
            return_tensors="pt",
            padding=False
        )

        input_ids      = full_inputs["input_ids"].squeeze(0)
        attention_mask = full_inputs["attention_mask"].squeeze(0)
        pixel_values   = full_inputs["pixel_values"]

        # ── 只對 answer 部分計算 loss
        labels = torch.full_like(input_ids, -100)
        labels[prompt_len:] = input_ids[prompt_len:]

        result = {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
            "pixel_values":   pixel_values,
        }
        # image_grid_thw 是 Qwen2-VL 用來記錄圖片 patch 排列的 tensor
        if "image_grid_thw" in full_inputs:
            result["image_grid_thw"] = full_inputs["image_grid_thw"]

        return result

# ── 資料量：500 筆 ────────────────────────────────────────
raw_dataset = load_dataset("HuggingFaceM4/ChartQA", split="train[:500]")
dataset     = ChartQADataset(raw_dataset, processor)
print(f"Dataset size: {len(dataset)}")

# ── 5. Custom Data Collator ──────────────────────────────
def data_collator(features):
    tokenizer = processor.tokenizer

    input_ids      = [f["input_ids"]      for f in features]
    attention_mask = [f["attention_mask"] for f in features]
    labels         = [f["labels"]         for f in features]

    batch = tokenizer.pad(
        {"input_ids": input_ids, "attention_mask": attention_mask},
        padding=True,
        return_tensors="pt"
    )

    max_len = batch["input_ids"].shape[1]
    padded_labels = []
    for l in labels:
        pad_len = max_len - l.shape[0]
        padded  = torch.cat([l, torch.full((pad_len,), -100, dtype=l.dtype)])
        padded_labels.append(padded)
    batch["labels"] = torch.stack(padded_labels)

    # pixel_values：不同圖片的 patch 數不同，沿 dim=0 cat
    if "pixel_values" in features[0]:
        batch["pixel_values"] = torch.cat(
            [f["pixel_values"] for f in features], dim=0
        )
    if "image_grid_thw" in features[0]:
        batch["image_grid_thw"] = torch.cat(
            [f["image_grid_thw"] for f in features], dim=0
        )

    return batch

# ── 6. Loss Callback ─────────────────────────────────────
loss_history = []

class LossLoggerCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            loss_history.append((state.global_step, logs['loss']))
            print(f"Step: {state.global_step} | Loss: {logs['loss']:.6f}")

# ── 7. Training ──────────────────────────────────────────
training_args = TrainingArguments(
    output_dir="./qwen2vl-finetuned",
    # ⚠️ Qwen2-VL 圖片 patch 數量不固定，batch_size=1 避免 OOM
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,   # effective batch = 16
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    fp16=True,
    logging_steps=1,
    num_train_epochs=3,
    save_strategy="no",
    remove_unused_columns=False,
    dataloader_num_workers=0,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
    callbacks=[LossLoggerCallback()]
)
trainer.train()

# ── 8. Loss Curve ─────────────────────────────────────────
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

steps  = [x[0] for x in loss_history]
losses = [x[1] for x in loss_history]

plt.figure(figsize=(10, 5))
plt.plot(steps, losses, marker='o', markersize=2)
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Training Loss Curve (QLoRA Fine-tuning on ChartQA, Qwen2-VL-7B)")
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_curve.png", dpi=150)
print("Loss curve saved to loss_curve.png")

# ── 9. Save Adapter ──────────────────────────────────────
print("Saving adapter...")
model.save_pretrained("./qwen2vl-finetuned")
processor.save_pretrained("./qwen2vl-finetuned")
print("Done! Adapter saved to ./qwen2vl-finetuned")
