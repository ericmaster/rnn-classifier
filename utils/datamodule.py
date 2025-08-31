import os
import glob
import random
import string
import unicodedata
from io import open
import torch
import pytorch_lightning as pl
import numpy as np
from torch.utils.data import Dataset, DataLoader
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


def randomTrainingExample(category_lines, all_categories):
    """Generate a random training example"""
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor


def collate_fn(batch):
    """Custom collate function to handle variable-length sequences"""
    labels, sequences = zip(*batch)
    
    # Stack labels
    labels = torch.stack(labels)
    
    # Pad sequences to the same length
    # sequences is a list of tensors with shape (seq_len, 1, n_letters)
    # We need to squeeze the middle dimension and then pad
    sequences = [seq.squeeze(1) for seq in sequences]  # Remove the 1 dimension
    padded_sequences = pad_sequence(sequences, batch_first=True)  # (batch_size, max_seq_len, n_letters)
    
    return labels, padded_sequences

class RNNDataset(Dataset):
    """Dataset con weighted sampling para balancear clases"""
    
    def __init__(self, category_lines, all_categories, num_samples=1000):
        self.category_lines = category_lines
        self.all_categories = all_categories
        self.num_samples = num_samples
        
        # Calcular pesos para cada categoría
        category_counts = [len(category_lines[cat]) for cat in all_categories]
        total_samples = sum(category_counts)
        
        # Pesos inversamente proporcionales al tamaño
        self.weights = [total_samples / (len(all_categories) * count) 
                        for count in category_counts]
        
        # Normalizar pesos
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]

        print(f"Pesos por categoría:")
        for cat, weight in zip(all_categories, self.weights):
            print(f"  {cat}: {weight:.4f}")
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Seleccionar categoría según los pesos
        category = np.random.choice(self.all_categories, p=self.weights)
        
        # Seleccionar nombre aleatorio de esa categoría
        line = randomChoice(self.category_lines[category])
        
        # Preparar tensores
        category_tensor = torch.tensor(self.all_categories.index(category), dtype=torch.long)
        line_tensor = lineToTensor(line)  # Shape: (seq_len, 1, n_letters)
        
        return category_tensor, line_tensor

class RNNDataModule(pl.LightningDataModule):
    """DataModule"""
    
    def __init__(self, data_path='./data/names/', batch_size=1):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        
        # Will be set in setup()
        self.category_lines = {}
        self.train_category_lines = {}
        self.val_category_lines = {}
        self.test_category_lines = {}
        self.all_categories = []
        self.n_categories = 0
        
    def setup(self, stage=None):
        """Setup data"""
        
        # Read all name files
        for filename in glob.glob(os.path.join(self.data_path, '*.txt')):
            category = os.path.splitext(os.path.basename(filename))[0]
            self.all_categories.append(category)
            lines = readLines(filename)
            self.category_lines[category] = lines
        
        self.n_categories = len(self.all_categories)
        print(f"Numero de clases: {self.n_categories}")

        # Split in train, val and test (80%, 10%, 10%)
        random.seed(42)  # For reproducibility
        for category in self.all_categories:
            lines = self.category_lines[category]
            random.shuffle(lines)
            n_total = len(lines)
            n_train = int(0.8 * n_total)
            n_val = int(0.1 * n_total)
            self.train_category_lines[category] = lines[:n_train]
            self.val_category_lines[category] = lines[n_train:n_train + n_val]
            self.test_category_lines[category] = lines[n_train + n_val:]
    
    def train_dataloader(self):
        """Return a generator that yields random examples"""
        dataset = RNNDataset(self.train_category_lines, self.all_categories, num_samples=100000)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn)
    
    def val_dataloader(self):
        """Return a generator for validation"""
        dataset = RNNDataset(self.val_category_lines, self.all_categories, num_samples=1000)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)
    
    def test_dataloader(self):
        """Return a generator for testing"""
        dataset = RNNDataset(self.test_category_lines, self.all_categories, num_samples=1000)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)
    
    def get_categories(self):
        """Get list of all categories"""
        return self.all_categories
    
    def get_n_categories(self):
        """Get number of categories"""
        return self.n_categories
