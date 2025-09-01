import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import numpy as np
from pytorch_lightning.loggers import CSVLogger

from .datamodule import lineToTensor

def evaluate_model(trainer, model, data_module):
    """
    Evaluate the trained model
    
    Args:
        model: Trained RNN classifier
        data_module: Data module
        n_samples: Number of samples to evaluate
    
    Returns:
        Accuracy and confusion matrix
    """
    base_model = model.base_model
    print(f"\nEvaluando modelo: {base_model.upper()}")

    logger = CSVLogger(save_dir="logs/rnn-classifier", name=f"{base_model}", version="eval")

    trainer = pl.Trainer(
        max_epochs=10,
        callbacks=[],
        accelerator="auto",  # Uses GPUs or TPUs if available
        devices="auto",  # Uses all available GPUs/TPUs if applicable
        logger=logger,
        deterministic=False,
        log_every_n_steps=10,
    )

    trainer.test(model, datamodule=data_module)

    cm = model.get_confusion_matrix()
    if cm is None:
        print(f"No se pudo obtener la matriz de confusión para el modelo {base_model}.")
        return model

    # # Visualizar matriz de confusión
    # plot_confusion_matrix(cm, categories, 'Matriz de Confusión - RNN Classifier')

    return model


def plot_confusion_matrix(cm, categories, title='Confusion Matrix'):
    """
    Plot confusion matrix
    
    Args:
        confusion: Confusion matrix tensor
        categories: List of category names
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    cax = ax.matshow(cm.cpu().numpy(), cmap='Blues')
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


def predict_name_origin(model, name, categories, n_predictions=3, verbose=True):
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
        
        # Obtenemos las N mejores categorias de clasificacion
        topv, topi = output.topk(n_predictions, 1, True)
        predictions = []
        
        for i in range(n_predictions):
            value = topv[0][i].item()
            category_index = topi[0][i].item()
            if verbose:
                print(f'({value:.2f}) {categories[category_index]}')
            predictions.append([value, categories[category_index]])
    
    return predictions