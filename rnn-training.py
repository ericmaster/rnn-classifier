import torch
import pytorch_lightning as pl
import time
import os

# Importar módulos personalizados
from utils.model import RNNClassifier, find_latest_checkpoint
from utils.datamodule import RNNDataset, RNNDataModule, n_letters

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
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
        "learning_rate": 0.00125,
        "n_epochs": 100,
        "batch_size": 32,
        "num_workers": 40,
    }

    print("Configuración del entrenamiento:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")

    base_models = ["rnn", "lstm", "gru"]
    # base_models = ["rnn"] # Para debug, solo entrenamos el modelo RNN

    for base_model in base_models:
        print(f"Probando modelo base: {base_model}")

        # Crear la instancia del modelo RNN
        print(f"Creando instancia de modelo RNN ({base_model})...")

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
            base_model=base_model,
            learning_rate=CONFIG["learning_rate"],
        )

        print(f"Modelo creado exitosamente!")
        print(f"Arquitectura del modelo:")
        print(f"  - Input size: {n_letters} (caracteres)")
        print(f"  - Hidden size: {CONFIG['hidden_size']}")
        print(f"  - Output size: {len(categories)} (categorías)")
        print(f"  - Learning rate: {CONFIG['learning_rate']}")

        # Guardamos el mejor modelo monitoreado en la acc de validación.
        callback_check = ModelCheckpoint(
            dirpath=f"checkpoints/{base_model}",
            save_top_k=1,
            mode="max",
            monitor="valid_acc"
        )  
        callback_tqdm = RichProgressBar(leave=True)
        callback_early_stop = EarlyStopping(
            monitor='valid_loss',
            min_delta=0.001,
            patience=10,
            verbose=True,
            mode='min'
        )
        logger = CSVLogger(save_dir="logs/rnn-classifier", name=f"{base_model}")

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
        # ckpt_path = f"./checkpoints/{base_model}/epoch=75-step=73188.ckpt"
        ckpt_path = find_latest_checkpoint(base_model)
        if (ckpt_path):
            print(f"Cargando modelo desde checkpoint: {ckpt_path}")
        else:
            print(f"No se encontró checkpoint en {ckpt_path}, entrenando desde cero.")

        trainer.fit(model=model, datamodule=data_module, ckpt_path=ckpt_path)

        runtime = (time.time() - start_time) / 60
        print(f"Tiempo de entrenamiento en minutos: {runtime:.2f}")

        print(f"¡Entrenamiento completado para RNN ({base_model})!")
