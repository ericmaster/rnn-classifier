import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import numpy as np
import time
import math

from .datamodule import lineToTensor, randomTrainingExample


def train_rnn(model, data_module, n_iters=10000):
    """
    Train RNN using manual optimization
    
    Args:
        model: Pre-instantiated RNNClassifier model
        data_path: Path to the names data directory
        n_iters: Number of training iterations
    
    Returns:
        Trained model and data module
    """
    
    print(f"Number of categories: {data_module.get_n_categories()}")
    print(f"Categories: {data_module.get_categories()}")
    
    # Set model to training mode
    model.train()
    
    # Training variables
    current_loss = 0
    all_losses = []
    print_every = 5000
    plot_every = 500
    
    def timeSince(since):
        now = time.time()
        s = now - since
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)
    
    start = time.time()
    
    print("Starting training...")
    
    # Training loop
    for iter in range(1, n_iters + 1):
        # Get random training example
        category, line, category_tensor, line_tensor = randomTrainingExample(
            data_module.category_lines, data_module.all_categories
        )
        
        # Move to device
        category_tensor = category_tensor.to(model.device)
        line_tensor = line_tensor.to(model.device)
        
        # Forward pass
        hidden = model.initHidden()
        model.zero_grad()
        
        for i in range(line_tensor.size(0)):
            output, hidden = model(line_tensor[i], hidden)
        
        # Calculate loss
        loss = model.criterion(output, category_tensor)
        
        # Manual backward and update
        model.manual_backward_step(loss)
        
        current_loss += loss.item()
        
        # Print progress
        if iter % print_every == 0:
            guess, guess_i = model.categoryFromOutput(output, data_module.all_categories)
            correct = '✓' if guess == category else '✗ (%s)' % category
            print('%d %d%% (%s) %.4f %s / %s %s' % (
                iter, iter / n_iters * 100, timeSince(start), 
                loss.item(), line, guess, correct
            ))
        
        # Add current loss avg to list of losses
        if iter % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0
    
    print("Training completed!")
    
    # Plot training progress
    plt.figure()
    plt.plot(all_losses)
    plt.title('Training Loss')
    plt.xlabel('Iterations (x100)')
    plt.ylabel('Loss')
    plt.show()
    
    return model, data_module


def evaluate_model(model, data_module, n_samples=1000):
    """
    Evaluate the trained model
    
    Args:
        model: Trained RNN classifier
        data_module: Data module
        n_samples: Number of samples to evaluate
    
    Returns:
        Accuracy and confusion matrix
    """
    model.eval()
    
    categories = data_module.get_categories()
    n_categories = len(categories)
    
    # Initialize confusion matrix
    confusion = torch.zeros(n_categories, n_categories)
    correct = 0
    total = 0
    
    with torch.no_grad():
        for i in range(n_samples):
            # Get random example
            category, line, category_tensor, line_tensor = randomTrainingExample(
                data_module.category_lines, data_module.all_categories
            )
            
            # Move to device
            line_tensor = line_tensor.to(model.device)
            
            # Evaluate
            output = model.evaluate_name(line_tensor)
            guess, guess_i = model.categoryFromOutput(output, categories)
            category_i = categories.index(category)
            
            # Update confusion matrix
            confusion[category_i][guess_i] += 1
            
            # Update accuracy
            correct += (guess == category)
            total += 1
    
    # Calculate accuracy
    accuracy = correct / total
    
    # Normalize confusion matrix
    for i in range(n_categories):
        if confusion[i].sum() > 0:
            confusion[i] = confusion[i] / confusion[i].sum()
    
    return accuracy, confusion, categories


def plot_confusion_matrix(confusion, categories, title='Confusion Matrix'):
    """
    Plot confusion matrix
    
    Args:
        confusion: Confusion matrix tensor
        categories: List of category names
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    cax = ax.matshow(confusion.numpy(), cmap='Blues')
    fig.colorbar(cax)
    
    # Set labels
    ax.set_xticklabels([''] + categories, rotation=90)
    ax.set_yticklabels([''] + categories)
    
    # Set ticks
    ax.xaxis.set_major_locator(plt.MultipleLocator(1))
    ax.yaxis.set_major_locator(plt.MultipleLocator(1))
    
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()


def predict_name_origin(model, name, categories, n_predictions=3):
    """
    Predict the origin of a name
    
    Args:
        model: Trained RNN classifier
        name: Name to classify
        categories: List of category names
        n_predictions: Number of top predictions to return
    
    Returns:
        List of predictions with probabilities
    """
    print(f'\n> {name}')
    
    model.eval()
    with torch.no_grad():
        line_tensor = lineToTensor(name).to(model.device)
        output = model.evaluate_name(line_tensor)
        
        # Get top N categories
        topv, topi = output.topk(n_predictions, 1, True)
        predictions = []
        
        for i in range(n_predictions):
            value = topv[0][i].item()
            category_index = topi[0][i].item()
            print(f'({value:.2f}) {categories[category_index]}')
            predictions.append([value, categories[category_index]])
    
    return predictions