import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class RNNClassifier(pl.LightningModule):
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.005):
        super(RNNClassifier, self).__init__()
        
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        
        # RNN layers
        self.i2h = nn.Linear(input_size, hidden_size)  # Wxh
        self.h2h = nn.Linear(hidden_size, hidden_size)  # Whh
        self.h2o = nn.Linear(hidden_size, output_size)  # Why
        self.softmax = nn.LogSoftmax(dim=1)
        
        # Loss function
        self.criterion = nn.NLLLoss()
        
        # Save hyperparameters
        self.save_hyperparameters()
    
    def forward(self, input, hidden):
        """Forward pass RNN"""
        hidden = F.tanh(self.i2h(input) + self.h2h(hidden))  # Wxh*x + Whh*h_(t-1)
        output = self.h2o(hidden)
        output = self.softmax(output)
        return output, hidden
    
    def initHidden(self):
        """Initialize hidden state"""
        return torch.zeros(1, self.hidden_size, device=self.device)
    
    def training_step(self, batch, batch_idx):
        """Training step"""
        category_tensor, line_tensor = batch
        
        # Process the sequence
        hidden = self.initHidden()
        
        for i in range(line_tensor.size(0)):
            output, hidden = self(line_tensor[i], hidden)
        
        # Calculate loss
        loss = self.criterion(output, category_tensor)
        
        # Log the loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step"""
        category_tensor, line_tensor = batch
        
        # Process the sequence
        hidden = self.initHidden()
        
        for i in range(line_tensor.size(0)):
            output, hidden = self(line_tensor[i], hidden)
        
        # Calculate loss
        loss = self.criterion(output, category_tensor)
        
        # Calculate accuracy
        _, predicted = torch.max(output, 1)
        accuracy = (predicted == category_tensor).float()
        
        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_accuracy', accuracy, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        """Use manual optimization"""
        # Return None to use manual optimization
        return None
    
    def manual_backward_step(self, loss):
        """Manual backward step"""
        loss.backward()
        
        # Manual parameter update
        with torch.no_grad():
            for p in self.parameters():
                if p.grad is not None:
                    p.data.add_(p.grad.data, alpha=-self.learning_rate)
        
        # Zero gradients
        self.zero_grad()
    
    def categoryFromOutput(self, output, all_categories):
        """Get category from output"""
        top_n, top_i = output.topk(1)
        category_i = top_i[0].item()
        return all_categories[category_i], category_i
    
    def evaluate_name(self, line_tensor):
        """Evaluate a single name"""
        hidden = self.initHidden()
        
        for i in range(line_tensor.size(0)):
            output, hidden = self(line_tensor[i], hidden)
        
        return output