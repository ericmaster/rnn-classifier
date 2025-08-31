import torch
import pytorch_lightning as pl
import time

# Importar módulos personalizados
from utils.model import RNNClassifier
from utils.datamodule import RNNDataset, RNNDataModule, n_letters

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import TQDMProgressBar, RichProgressBar

if __name__ == "__main__":
    torch.manual_seed(47)

    print(f"PyTorch version: {torch.__version__}")
    print(f"PyTorch Lightning version: {pl.__version__}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    # Configuración del entrenamiento
    CONFIG = {
        "hidden_size": 128,
        "learning_rate": 0.005,
        "n_epochs": 1,
        "batch_size": 64,
        "num_workers": 40,
    }

    print("Configuración del entrenamiento:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")

    # Crear la instancia del modelo RNN
    print("Creando modelo RNN...")

    data_module = RNNDataModule(
        data_path="./data/names/", batch_size=CONFIG["batch_size"], num_workers=40
    )
    data_module.setup()
    categories = data_module.dataset.get_categories()

    # Crear el modelo con la configuración especificada
    model = RNNClassifier(
        input_size=n_letters,
        hidden_size=CONFIG["hidden_size"],
        output_size=len(categories),
        learning_rate=CONFIG["learning_rate"],
    )

    print(f"Modelo creado exitosamente!")
    print(f"Arquitectura del modelo:")
    print(f"  - Input size: {n_letters} (caracteres)")
    print(f"  - Hidden size: {CONFIG['hidden_size']}")
    print(f"  - Output size: {len(categories)} (categorías)")
    print(f"  - Learning rate: {CONFIG['learning_rate']}")

    # Inicialización del modulo lightning
    callback_check = ModelCheckpoint(
        save_top_k=1, mode="max", monitor="valid_acc"
    )  # guardamos el mejor modelo monitoreado en la acc de validación.
    callback_tqdm = RichProgressBar(leave=True)
    logger = CSVLogger(save_dir="logs/", name="rnn-classifier")

    # Inicia entrenamiento
    trainer = pl.Trainer(
        max_epochs=CONFIG["n_epochs"],
        callbacks=[callback_check, callback_tqdm],
        accelerator="auto",  # Uses GPUs or TPUs if available
        devices="auto",  # Uses all available GPUs/TPUs if applicable
        logger=logger,
        deterministic=False,
        log_every_n_steps=10,
    )

    start_time = time.time()
    trainer.fit(model=model, datamodule=data_module)

    runtime = (time.time() - start_time) / 60
    print(f"Tiempo de entrenamiento en minutos: {runtime:.2f}")

    print("¡Entrenamiento completado!")
