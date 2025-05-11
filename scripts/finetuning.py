from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import torch
import os
import bitsandbytes as bnb

# Detectar si estamos en Google Colab
IN_COLAB = 'google.colab' in str(get_ipython())

# === CONFIGURACIÓN ===
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
DATASET_PATH = "./data/reglamento.json"
OUTPUT_DIR = "./models/mistral-lora-output"

# Crear directorio de salida si no existe
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === TOKENIZER ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token  # Para evitar errores de padding

# === MODELO ===
if IN_COLAB or torch.cuda.is_available():
    # En entornos con GPU, usar quantización 8-bit para reducir uso de memoria
    print("Usando GPU con quantización 8-bit...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        load_in_8bit=True,
        device_map="auto",
    )
else:
    # En entornos sin GPU, usar float32
    print("Usando CPU con precisión completa (float32)...")
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
    gradient_accumulation_steps=4 if (IN_COLAB or torch.cuda.is_available()) else 1,
    logging_steps=10,
    save_strategy="epoch",
    learning_rate=2e-4,
    fp16=True if (IN_COLAB or torch.cuda.is_available()) else False,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

trainer.train()
