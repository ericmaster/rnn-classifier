import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy


class RNNClassifier(pl.LightningModule):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        base_model="rnn",
        learning_rate=0.005,
    ):
        super(RNNClassifier, self).__init__()

        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.output_size = output_size

        if (base_model == "rnn"):
            self.base_model = nn.RNN(input_size, hidden_size, num_layers=1, nonlinearity='tanh', batch_first=True)
        elif (base_model == "lstm"):
            self.base_model = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        elif (base_model == "gru"):
            self.base_model = nn.GRU(input_size, hidden_size, num_layers=1, batch_first=True)
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

        # Save hyperparameters
        self.save_hyperparameters()

    def forward(self, input):
        """Forward pass RNN"""

        # input shape: (batch_size, seq_len, input_size)
        batch_size = input.size(0)
        # hidden_0 shape: (num_layers, batch_size, hidden_size)
        hidden_0 = self.initHidden(batch_size)

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

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = self.learning_rate)
        return optimizer

    def initHidden(self, batch_size=1):
        """Initialize hidden state"""
        return torch.zeros(1, batch_size, self.hidden_size, dtype=torch.float32, device=self.device)

    def categoryFromOutput(self, output, all_categories):
        """Get category from output"""
        top_n, top_i = output.topk(1)
        category_i = top_i[0].item()
        return all_categories[category_i], category_i

    def evaluate_name(self, line_tensor):
        """Evaluate a single name"""
        # Add batch dimension if not present
        if line_tensor.dim() == 3 and line_tensor.size(0) == 1:
            # Already has batch dimension
            output = self(line_tensor)
        else:
            # Add batch dimension
            line_tensor = line_tensor.unsqueeze(0)  # Add batch dimension
            output = self(line_tensor)

        return output
