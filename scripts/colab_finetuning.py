from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import torch
import os

# === CONFIGURACIÓN ===
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
DATASET_PATH = "./data/reglamento.json"
OUTPUT_DIR = "./models/mistral-lora-output"

# Asegurarnos de que el directorio de salida exista
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === TOKENIZER ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token  # Para evitar errores de padding

# === MODELO ===
print("Cargando modelo base...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, 
    torch_dtype=torch.float16,  # Usar float16 para ahorrar memoria en Colab
    device_map="auto",          # Distribuir automáticamente en GPU disponibles
)
model.resize_token_embeddings(len(tokenizer))

# === CONFIGURACIÓN LoRA ===
print("Aplicando LoRA...")
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # Mostrar los parámetros entrenables

# === CARGAR DATASET ===
print("Cargando dataset...")
dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
print(f"Dataset cargado con {len(dataset)} ejemplos")

def tokenize(example):
    prompt = f"<s>[INST] {example['instruction']} [/INST] {example['output']} </s>"
    tokens = tokenizer(prompt, truncation=True, padding="max_length", max_length=1024)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

print("Tokenizando dataset...")
dataset = dataset.map(tokenize)

# === ENTRENAMIENTO ===
print("Configurando entrenamiento...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,  # Acumular gradientes para simular batch más grande
    logging_steps=10,
    save_strategy="epoch",
    learning_rate=2e-4,
    fp16=True,  # Usar precisión mixta para ahorrar memoria
    report_to="none",
    optim="adamw_torch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

print("Iniciando entrenamiento...")
trainer.train()

# Guardar modelo
print("Guardando modelo...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("¡Entrenamiento completado!")
