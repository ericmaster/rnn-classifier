import time
import torch
import torchaudio.transforms as T
import pytorch_lightning as pl

from crnn_utils.datamodule import EnviromentalDataset, EnviromentalDataModule
from crnn_utils.model import Lightning_CRNN
from torch.utils.data import DataLoader

from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import (
    RichProgressBar,
    EarlyStopping,
    LearningRateMonitor,
)

torch.manual_seed(47)
BATCH_SIZE = 64
NUM_EPOCHS = 200
LEARNING_RATE = 1e-4  # Lo suficientemente alto para converger rápido
NUM_WORKERS = 40
CLASES = 50
DROPOUT_RATE = 0.3
WEIGHT_DECAY = 1e-5

ANNOTATIONS_FILE = "./data/esc50/esc50.csv"
AUDIO_DIR = "./data/esc50/audio/audio/44100"
SAMPLE_RATE = 22050
MAX_LEN_SEC = 5

mel_transform = T.MelSpectrogram(
    sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=512, n_mels=64
)
env_dataset = EnviromentalDataset(
    ANNOTATIONS_FILE, AUDIO_DIR, mel_transform, SAMPLE_RATE, MAX_LEN_SEC
)
env_dataloader = DataLoader(
    env_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
)

# Tomar un batch para fijar dimensiones
sample_batch, _ = next(iter(env_dataloader))  # [B, 1, 64, T]
_, C, H, W = sample_batch.shape
print(f"Input mel shape: C={C}, H(n_mels)={H}, W(frames)={W}")

data_module = EnviromentalDataModule(
    ANNOTATIONS_FILE,
    AUDIO_DIR,
    mel_transform,
    SAMPLE_RATE,
    MAX_LEN_SEC,
    BATCH_SIZE,
    NUM_WORKERS,
)

input_dim = (C, H, W)
lightning_model = Lightning_CRNN(
    input_dim=input_dim,
    num_classes=CLASES,
    lr=LEARNING_RATE,
    dropout_rate=DROPOUT_RATE,
    weight_decay=WEIGHT_DECAY,
)

callback_check = ModelCheckpoint(
    dirpath=f"checkpoints/crnn",
    save_top_k=1,
    verbose=True,
    monitor="valid_acc",
    mode="max",
)

early_stopping = EarlyStopping(
    monitor="valid_loss", patience=20, verbose=True, mode="min"
)

lr_monitor = LearningRateMonitor(logging_interval="epoch")

logger = CSVLogger(save_dir="logs/", name="crnn-esc50")

progress = RichProgressBar(leave=True)

trainer = pl.Trainer(
    max_epochs=NUM_EPOCHS,
    callbacks=[callback_check, early_stopping, lr_monitor, progress],
    accelerator="auto",
    devices="auto",
    logger=logger,
    deterministic=False,
    log_every_n_steps=10,
)

print("Configuración de entrenamiento:")
print(f"- Early stopping: {early_stopping.patience} épocas de paciencia")
print(f"- Checkpoint: Top-{callback_check.save_top_k} modelos guardados")
print(f"- Learning rate monitoring: Habilitado")
print(f"- Logger: {logger.name}")
print(f"- Callbacks: {len(trainer.callbacks)} callbacks")

# Ejecutar entrenamiento mejorado
print("Iniciando entrenamiento con data augmentations...")
print("=" * 60)

start = time.time()
trainer.fit(model=lightning_model, datamodule=data_module)
print(f"Minutos entrenamiento: {(time.time()-start)/60:.2f}")

print("\\n" + "=" * 60)
print("Entrenamiento completado!")
