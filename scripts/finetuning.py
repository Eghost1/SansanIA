from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training
import torch
import os

# === CONFIGURACIÓN ===
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
DATASET_PATH = "./data/reglamento.json"
OUTPUT_DIR = "./models/mistral-lora-output"

# === TOKENIZER ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token  # Para evitar errores de padding

# === MODELO ===
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
model.resize_token_embeddings(len(tokenizer))

# === CONFIGURACIÓN LoRA ===
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# === CARGAR DATASET ===
dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

def tokenize(example):
    prompt = f"<s>[INST] {example['instruction']} [/INST] {example['output']} </s>"
    tokens = tokenizer(prompt, truncation=True, padding="max_length", max_length=1024)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

dataset = dataset.map(tokenize)

# === ENTRENAMIENTO ===
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    logging_steps=10,
    save_strategy="epoch",
    learning_rate=2e-4,
    fp16=False,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

trainer.train()
