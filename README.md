# SansanIA - Fine-tuning de Mistral 7B

Este proyecto contiene scripts para fine-tuning del modelo Mistral-7B-Instruct-v0.3 utilizando LoRA (Low-Rank Adaptation) con datos de reglamento universitario.

## Estructura del proyecto

```
SansanIA/
├── data/               # Datos de entrenamiento
│   └── reglamento.json # Dataset de reglamentos universitarios
├── models/             # Modelos entrenados (generados durante el fine-tuning)
├── scripts/            # Scripts de entrenamiento
│   ├── finetuning.py          # Script de fine-tuning para entorno local
│   └── finetuning_colab.ipynb # Notebook para Google Colab
└── requirements.txt    # Dependencias del proyecto
```

## Requisitos

- Python 3.8+
- PyTorch 2.0+
- CUDA (para entrenamiento en GPU)

## Instalación

1. Clonar el repositorio:
   ```bash
   git clone https://github.com/tu-usuario/SansanIA.git
   cd SansanIA
   ```

2. Crear un entorno virtual e instalar dependencias:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # En Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

## Uso

### Entrenamiento local

Para ejecutar el fine-tuning en tu máquina local:

```bash
cd scripts
python finetuning.py
```

### Entrenamiento en Google Colab

1. Abrir el notebook `scripts/finetuning_colab.ipynb` en Google Colab
2. Elegir un runtime con GPU (Runtime > Change runtime type > Hardware accelerator > GPU)
3. Ejecutar todas las celdas en orden

El modelo entrenado se guardará automáticamente en Google Drive.

## Dataset

El dataset `reglamento.json` contiene preguntas y respuestas relacionadas con reglamentos universitarios, estructurado para fine-tuning de modelos conversacionales.

## Configuración

Los principales parámetros se pueden ajustar en los scripts:

- `MODEL_NAME`: Modelo base a utilizar
- `DATASET_PATH`: Ruta al dataset
- `OUTPUT_DIR`: Directorio donde se guardarán los modelos entrenados
- Hiperparámetros de LoRA (r, alpha, target_modules, etc.)
- Hiperparámetros de entrenamiento (epochs, batch_size, learning_rate, etc.)

## Licencia

[Incluir información de licencia]
