import os
import glob
import random
import string
import unicodedata
from io import open
import torch
import pytorch_lightning as pl
import numpy as np
import subprocess
import zipfile
import urllib.request
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence


# Global variables for character encoding
all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)


def unicodeToAscii(s):
    """Normalize words with accents"""
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)  # separa el acento de la letra
        if unicodedata.category(c) != 'Mn'  # "Mark, Nonspacing" elimina los acentos
        and c in all_letters
    )


def letterToIndex(letter):
    """Find the index of a letter"""
    return all_letters.find(letter)


def letterToTensor(letter):
    """Represent a letter as one-hot <1 x n_letters>"""
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor


def lineToTensor(line):
    """Represent a sample as one-hot <line_length x 1 x n_letters>"""
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1  # poner 1 en la posicion del index
    return tensor


def readLines(filename):
    """Read lines from a file and normalize them"""
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]


def randomChoice(l):
    """Random choice"""
    return l[random.randint(0, len(l) - 1)]

def download_data(data_path='./data'):
    """Download and extract the names dataset if it doesn't exist"""
    if os.path.exists(data_path):
        print("Los datos ya están disponibles.")
        return
    
    print("Descargando datos...")
    
    # Download the data
    url = "https://download.pytorch.org/tutorial/data.zip"
    zip_path = "data.zip"
    
    try:
        urllib.request.urlretrieve(url, zip_path)
        print("Descarga completada.")
        
        # Extract the zip file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall('.')
        
        # Remove the zip file
        os.remove(zip_path)
        print("Datos descargados y extraídos exitosamente!")
        
    except Exception as e:
        print(f"Error al descargar los datos: {e}")
        # Fallback to wget if urllib fails
        try:
            print("Intentando con wget...")
            subprocess.run(['wget', url], check=True)
            subprocess.run(['unzip', zip_path], check=True)
            subprocess.run(['rm', zip_path], check=True)
            print("Datos descargados exitosamente con wget!")
        except subprocess.CalledProcessError as e:
            print(f"Error con wget: {e}")
            raise Exception("No se pudieron descargar los datos")

def collate_fn(batch):
    """Custom collate function to handle variable-length sequences"""
    sequences, labels = zip(*batch)
    
    # Stack labels
    labels = torch.stack(labels)
    
    # Agregamos padding a las secuencias para que todas tengan la misma longitud
    # sequences: tensores con shape (seq_len, 1, n_letters)
    # Hacemos squeeze para eliminar la dimension 1 y luego aplicamos pad_sequence
    sequences = [seq.squeeze(1) for seq in sequences]
    sequences = pad_sequence(sequences, batch_first=True)  # (batch_size, max_seq_len, n_letters)
    
    return sequences, labels

class RNNDataset(Dataset):
    """Dataset con weighted sampling para balancear clases"""
    
    def __init__(self, data_path='./data/names/', balanced=True, verbose=True, auto_download=True):
        self.data_path = data_path
        self.categories = []
        self.lines = []
        self.X = []
        self.y = []
        self.category_lines = {}
        self.category_X = {}
        self.category_y = {}

        # Descargar datos automáticamente si no existen
        if auto_download:
            download_data('./data')

        # Leemos todos los archivos y almacenamos las lineas por categoria
        if (verbose):
            print(f"Cargando datos desde {data_path}...")
        for filename in glob.glob(os.path.join(self.data_path, '*.txt')):
            category = os.path.splitext(os.path.basename(filename))[0]
            self.categories.append(category)
            category_tensor = torch.tensor(self.categories.index(category), dtype=torch.long)
            lines = readLines(filename)
            self.category_lines[category] = []
            self.category_X[category] = []
            self.category_y[category] = []
            for line in lines:
                self.lines.append(line)
                self.X.append(lineToTensor(line))
                self.y.append(category_tensor)
                self.category_lines[category].append(line)
                self.category_X[category].append(lineToTensor(line))
                self.category_y[category].append(category_tensor)

        # Imprimimos el numero de items por categoria
        if (verbose):
            for cat in self.categories:
                print(f"{cat}: {len(self.category_lines[cat])} ejemplos")

        # Incrementamos el dataset para balancear las clases
        if balanced:
            if (verbose):
                print()
                print(f"Balanceando dataset...")
            max_size = max([self.y.count(torch.tensor([i])) for i in range(len(self.categories))])
            new_X = []
            new_y = []
            new_lines = []
            for cat in self.categories:
                # Si la categoria ya tiene el tamaño maximo, no hacemos nada
                if len(self.category_X[cat]) >= max_size:
                    new_X.extend(self.category_X[cat])
                    new_y.extend(self.category_y[cat])
                    new_lines.extend(self.category_lines[cat])
                    if (verbose):
                        print(f"{cat} - tamaño original: {len(self.category_X[cat])}")
                    continue
                # TODO: mejorar esto para que no sean copias exactas. Generar con LLM?
                new_category_X = self.category_X[cat] * (max_size // len(self.category_X[cat]))
                new_category_Y = self.category_y[cat] * (max_size // len(self.category_X[cat]))
                new_category_lines = self.category_lines[cat] * (max_size // len(self.category_X[cat]))
                new_X.extend(new_category_X)
                new_y.extend(new_category_Y)
                new_lines.extend(new_category_lines)
                if (verbose):
                    print(f"{cat} - nuevo tamaño: {len(new_category_X)}")
            self.X = new_X
            self.y = new_y
            self.lines = new_lines
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def get_categories(self):
        """Get list of all categories"""
        return self.categories
    
    def get_random_sample(self):
        """Get a random sample from the dataset"""
        idx = random.randint(0, len(self.y) - 1)
        category = self.categories[self.y[idx].item()]
        line_tensor = self.X[idx]
        line = self.lines[idx]
        category_tensor = self.y[idx]
        return category, line, category_tensor, line_tensor

class RNNDataModule(pl.LightningDataModule):
    """DataModule"""
    
    def __init__(self, data_path='./data/names/', batch_size=64, num_workers=4, auto_download=True):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.auto_download = auto_download
        self.dataset = None
        
    def setup(self, stage=None):
        """Setup data"""
        
        self.dataset = RNNDataset(data_path=self.data_path, balanced=True, verbose=False, auto_download=self.auto_download)
        
        # Split into train, validation, test

        train_size = int(0.8 * len(self.dataset))
        val_test_size = len(self.dataset) - train_size
        val_size = val_test_size // 2
        test_size = val_test_size - val_size

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(self.dataset, [train_size, val_size, test_size])
    
    def train_dataloader(self):
        """Return a generator that yields random examples"""
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=self.num_workers)
    
    def val_dataloader(self):
        """Return a generator for validation"""
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=self.num_workers)
    
    def test_dataloader(self):
        """Return a generator for testing"""
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=self.num_workers)

