import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import numpy as np
import time
import math

from .datamodule import lineToTensor

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