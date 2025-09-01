import os
import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pytorch_lightning.loggers import CSVLogger

from .datamodule import lineToTensor

def evaluate_model(model, data_module):
    """
    Evaluate the trained model
    
    Args:
        model: Trained RNN classifier
        data_module: Data module
        n_samples: Number of samples to evaluate
    
    Returns:
        Accuracy and confusion matrix
    """

    base_model_type = model.base_model_type
    print(f"\nEvaluando modelo: {base_model_type.upper()}")

    logger = CSVLogger(save_dir="logs/rnn-classifier", name=f"{base_model_type}", version="eval")

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

    return model

def plot_metrics(base_model):
    # Concatenate metric files from all versions for the given model
    log_dir = f"logs/rnn-classifier/{base_model}"
    all_metrics = []

    if not os.path.exists(log_dir):
        print(f"No se encontró el directorio de logs para el modelo {base_model}. Saltando visualización.")
        return
    for version in os.listdir(log_dir):
        if not version.startswith("version_"):
            continue
        version_dir = os.path.join(log_dir, version)
        metrics_file = os.path.join(version_dir, "metrics.csv")
        if os.path.exists(metrics_file):
            df = pd.read_csv(metrics_file)
            all_metrics.append(df)

    if not all_metrics:
        print(f"No se encontraron archivos de métricas para el modelo {base_model}. Saltando visualización.")
        return

    metrics = pd.concat(all_metrics, ignore_index=True)

    aggreg_metrics = []
    agg_col = "epoch"
    for i, dfg in metrics.groupby(agg_col):
        agg = dict(dfg.mean())
        agg[agg_col] = i
        aggreg_metrics.append(agg)

    df_metrics = pd.DataFrame(aggreg_metrics)
    df_metrics[["train_loss", "valid_loss"]].plot(
        grid=True, legend=True, xlabel="Epoch", ylabel="Loss", title=f"Loss en RNN con {base_model.upper()}"
    )
    df_metrics[["train_acc", "valid_acc"]].plot(
        grid=True, legend=True, xlabel="Epoch", ylabel="Accuracy", title=f"Accuracy en RNN con {base_model.upper()}"
    )

    plt.show()

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
    if verbose:
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