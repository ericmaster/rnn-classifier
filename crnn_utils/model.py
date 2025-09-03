import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import pytorch_lightning as pl

class CRNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        """
        input_dim: (C=1, H=64, W=frames)
        Conv + Pool x2 -> H' = H//4, W' = W//4, C' = 32
        LSTM: seq_len = W', input_size = 32 * H'
        """
        super().__init__()
        C, H, W = input_dim
        self.conv1 = nn.Conv2d(C, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()

        H2 = H // 4
        W2 = W // 4
        self.lstm_input_size = 32 * H2
        self.lstm = nn.LSTM(input_size=self.lstm_input_size, hidden_size=256, num_layers=2,
                            batch_first=True, bidirectional=True)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):               # x: [B, 1, 64, W]
        x = self.pool(self.relu(self.conv1(x)))  # [B,16,H/2,W/2]
        x = self.pool(self.relu(self.conv2(x)))  # [B,32,H/4,W/4]
        # Reorganizar: secuencia = eje temporal (W/4)
        B, C, H2, W2 = x.shape
        x = x.permute(0, 3, 1, 2).contiguous()   # [B, W2, C, H2]
        x = x.view(B, W2, C * H2)                # [B, W2, 32*H2]
        x, _ = self.lstm(x)                      # [B, W2, 512]
        x = x[:, -1, :]                          # último paso temporal
        x = self.fc(x)                           # logits [B, num_classes]
        return x                                 # NO softmax

class Lightning_CRNN(pl.LightningModule):
    def __init__(self, input_dim, num_classes, lr):
        super().__init__()
        self.save_hyperparameters()
        self.model = CRNN(input_dim=input_dim, num_classes=num_classes)
        self.train_acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
        self.valid_acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
        self.test_acc  = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch):
        x, y = batch
        logits = self(x)                                # logits
        loss = F.cross_entropy(logits, y)               # espera logits
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch, batch_idx):
        loss, preds, y = self._shared_step(batch)
        self.train_acc.update(preds, y)
        self.log("train_loss", loss, prog_bar=False)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, y = self._shared_step(batch)
        self.valid_acc.update(preds, y)
        self.log("valid_loss", loss, prog_bar=False)
        self.log("valid_acc", self.valid_acc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, preds, y = self._shared_step(batch)
        self.test_acc.update(preds, y)
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_train_epoch_end(self):
        self.train_acc.reset()

    def on_validation_epoch_end(self):
        self.valid_acc.reset()

    def on_test_epoch_end(self):
        self.test_acc.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
