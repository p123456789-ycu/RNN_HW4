import torch
import sys
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset
from qwen_vl_utils import process_vision_info

model_id     = "Qwen/Qwen2-VL-7B-Instruct"
adapter_path = "./qwen2vl-finetuned"

# ── Quantization Config ──────────────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

def run_inference(use_adapter: bool):
    tag      = "True" if use_adapter else "False"
    log_name = f"log_inference_{tag}.txt"

    original_stdout = sys.__stdout__
    log_file = open(log_name, "w", encoding="utf-8")

    class TeeWriter:
        def write(self, msg):
            original_stdout.write(msg)
            log_file.write(msg)
        def flush(self):
            original_stdout.flush()
            log_file.flush()
        def isatty(self):
            return False

    sys.stdout = TeeWriter()

    print(f"{'='*50}")
    print(f"Running inference | use_adapter={use_adapter} | log={log_name}")
    print(f"{'='*50}\n")

    # ── Load Model ───────────────────────────────────────
    print("Loading Qwen2-VL-7B-Instruct in 4-bit...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_id)
    print("Base model loaded.")

    # ── Load Adapter ─────────────────────────────────────
    if use_adapter:
        model = PeftModel.from_pretrained(model, adapter_path)
        print("LoRA adapter loaded.")

    model.eval()

    # ── Load ChartQA Test Data ────────────────────────────
    print("\n載入 ChartQA 測試資料...")
    test_data = load_dataset("HuggingFaceM4/ChartQA", split="test")

    # ── Inference on 5 samples ────────────────────────────
    for i in range(10):
        sample       = test_data[i]
        image        = sample['image'].convert("RGB")
        question     = sample['query']
        ground_truth = sample['label'][0]

        # ── Qwen2-VL messages 格式（跟訓練時完全一致）
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text",  "text": question}
                ]
            }
        ]

        prompt_text  = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, _ = process_vision_info(messages)
        inputs = processor(
            text=[prompt_text],
            images=image_inputs,
            return_tensors="pt"
        ).to("cuda")

        with torch.no_grad():
            generate_ids = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False
            )

        # ── 只 decode 新生成的 token
        new_tokens = generate_ids[:, inputs["input_ids"].shape[1]:]
        answer = processor.decode(
            new_tokens[0],
            skip_special_tokens=True
        ).strip()

        print(f"\n{'='*50}")
        print(f"[Sample {i+1}]")
        print(f"Question    : {question}")
        print(f"Ground Truth: {ground_truth}")
        print(f"Model Answer: {answer}")

    print("\nDone.")

    log_file.flush()
    log_file.close()
    sys.stdout = original_stdout
    print(f"[✓] Log saved to {log_name}\n")

    del model
    torch.cuda.empty_cache()


# ── 分別跑 Before 和 After ────────────────────────────────
run_inference(use_adapter=False)   # Part 1: Before
run_inference(use_adapter=True)    # Part 3: After
