import time
import torch
import torchaudio.transforms as T
import pytorch_lightning as pl

from crnn_utils.datamodule import EnviromentalDataset, EnviromentalDataModule
from crnn_utils.model import Lightning_CRNN
from torch.utils.data import DataLoader

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import RichProgressBar

torch.manual_seed(47)
BATCH_SIZE = 32
NUM_EPOCHS = 40
LEARNING_RATE = 1e-4
NUM_WORKERS = 4
CLASES = 50

ANNOTATIONS_FILE = "./data/esc50/esc50.csv"
AUDIO_DIR = "./data/esc50/audio/audio/44100"
SAMPLE_RATE = 16_000
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
    input_dim=input_dim, num_classes=CLASES, lr=LEARNING_RATE
)

callback_check = ModelCheckpoint(
    dirpath=f"checkpoints/crnn", save_top_k=1, mode="max", monitor="valid_acc"
)
logger = CSVLogger(save_dir="logs/", name="crnn-esc50")
progress = RichProgressBar(leave=True)

trainer = pl.Trainer(
    max_epochs=NUM_EPOCHS,
    callbacks=[callback_check, progress],
    accelerator="auto",
    devices="auto",
    logger=logger,
    deterministic=False,
    log_every_n_steps=10,
)

start = time.time()
trainer.fit(model=lightning_model, datamodule=data_module)
print(f"Minutos entrenamiento: {(time.time()-start)/60:.2f}")
