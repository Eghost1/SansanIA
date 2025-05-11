import os
import subprocess

def configurar_entorno():
    print("Configurando entorno para entrenamiento en Google Colab...")
    
    # Crear directorios necesarios
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("models/mistral-lora-output", exist_ok=True)
    
    # Instalar dependencias
    print("Instalando dependencias...")
    subprocess.run("pip install -q torch>=2.0.0 transformers>=4.37.0 datasets>=2.14.0 peft>=0.4.0 accelerate>=0.23.0 scipy tqdm bitsandbytes>=0.39.0", shell=True)
    
    print("Entorno configurado correctamente.")
    
if __name__ == "__main__":
    configurar_entorno()
