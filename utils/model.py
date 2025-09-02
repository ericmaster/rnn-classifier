import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import os
import glob
from torchmetrics import Accuracy, ConfusionMatrix

def find_latest_checkpoint(base_model_name):
    checkpoints_dir = f"./checkpoints/{base_model_name}/"
    all_checkpoints = []
    for filename in glob.glob(os.path.join(checkpoints_dir, '*.ckpt')):
        all_checkpoints.append(filename)
    
    if all_checkpoints:
        return max(all_checkpoints, key=os.path.getmtime)
    return None


class RNNClassifier(pl.LightningModule):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        num_layers=1,
        base_model="rnn",
        learning_rate=0.005,
    ):
        super(RNNClassifier, self).__init__()

        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.output_size = output_size
        self.num_layers = num_layers
        self.base_model_type = base_model

        if (base_model == "rnn"):
            self.base_model = nn.RNN(input_size, hidden_size, num_layers=self.num_layers, nonlinearity='tanh', batch_first=True)
        elif (base_model == "lstm"):
            self.base_model = nn.LSTM(input_size, hidden_size, num_layers=self.num_layers, batch_first=True)
        elif (base_model == "gru"):
            self.base_model = nn.GRU(input_size, hidden_size, num_layers=self.num_layers, batch_first=True)
        else:
            raise ValueError(f"Unknown base_model: {base_model}")
        self.out = nn.Linear(hidden_size, output_size)

        self.softmax = nn.LogSoftmax(dim=1)

        # Loss function
        self.criterion = nn.NLLLoss()
        
        # Metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=output_size)
        self.valid_acc = Accuracy(task="multiclass", num_classes=output_size)
        self.test_acc = Accuracy(task="multiclass", num_classes=output_size)

        # Confusion matrix
        self.test_confmat = ConfusionMatrix(task="multiclass", num_classes=output_size)

        # Save hyperparameters
        self.save_hyperparameters()

    def forward(self, input):
        """Forward pass RNN"""

        # input shape: (batch_size, seq_len, input_size)
        batch_size = input.size(0)
        # Inicializamos el hidden state (incluye cell state para LSTM)
        hidden_0 = self.initHidden(batch_size)

        if self.base_model_type == "lstm":
            # LSTM recibe y devuelve (hidden_state, cell_state)
            out, (hidden_n, cell_n) = self.base_model(input, hidden_0)
        else:
            # RNN and GRU espera y devuelve hidden state
            out, hidden_n = self.base_model(input, hidden_0)
        # out shape: (batch_size, seq_len, hidden_size)
        # hidden_n shape: (num_layers, batch_size, hidden_size)

        # Usamos el ultimo output de la secuencia
        output = self.out(out[:, -1, :]) # (batch_size, output_size)
        output = self.softmax(output)
        return output

    # Pasos del proceso forward comunes entre train, val, test
    def _shared_step(self, batch):
        features, true_labels = batch
        probs = self(features)
        loss = self.criterion(probs, true_labels) # NLLloss receives logits
        predicted_labels = torch.argmax(probs, dim=1)

        return loss, true_labels, predicted_labels

    def training_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)
        self.log("train_loss", loss)
        self.train_acc(predicted_labels, true_labels)
        self.log("train_acc", self.train_acc, on_epoch=True, on_step=False)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)
        self.log("valid_loss", loss)
        self.valid_acc(predicted_labels, true_labels)
        self.log("valid_acc", self.valid_acc, on_epoch=True, on_step=False, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)
        self.test_acc(predicted_labels, true_labels)
        self.log("test_acc", self.test_acc, on_epoch=True, on_step=False)
        self.test_confmat.update(predicted_labels, true_labels)

    def on_test_epoch_end(self):
        self.test_confmat_result = self.test_confmat.compute()
        self.test_confmat.reset()

    def get_confusion_matrix(self):
        return self.test_confmat_result

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = self.learning_rate)
        return optimizer

    def initHidden(self, batch_size=1):
        """Initialize hidden state"""
        if self.base_model_type == "lstm":
            # LSTM necesita ambos hidden state y cell state
            hidden_state = torch.zeros(self.num_layers, batch_size, self.hidden_size, dtype=torch.float32, device=self.device)
            cell_state = torch.zeros(self.num_layers, batch_size, self.hidden_size, dtype=torch.float32, device=self.device)
            return (hidden_state, cell_state)
        else:
            # RNN y GRU solo necesitan hidden state
            return torch.zeros(self.num_layers, batch_size, self.hidden_size, dtype=torch.float32, device=self.device)

    def categoryFromOutput(self, output, all_categories):
        """Get category from output"""
        top_n, top_i = output.topk(1)
        category_i = top_i[0].item()
        return all_categories[category_i], category_i

    def evaluate_name(self, line_tensor):
        """Evaluate a single name"""

        if line_tensor.dim() == 3:
            # lineToTensor shape: (seq_len, 1, n_letters)
            # Convertimos dimensiones: (1, seq_len, n_letters) para batch_first=True
            line_tensor = line_tensor.transpose(0, 1)  # (1, seq_len, n_letters)
        elif line_tensor.dim() == 2:
            # La linea tiene el formato (seq_len, n_letters), solo agregamos la dimension de batch
            line_tensor = line_tensor.unsqueeze(0)  # (1, seq_len, n_letters)
        
        output = self(line_tensor)
        return output
