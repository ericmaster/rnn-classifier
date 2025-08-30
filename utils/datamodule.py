import os
import glob
import random
import string
import unicodedata
from io import open
import torch
import pytorch_lightning as pl


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


class RNNDataModule(pl.LightningDataModule):
    """DataModule"""
    
    def __init__(self, data_path='./data/names/', batch_size=1):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        
        # Will be set in setup()
        self.category_lines = {}
        self.all_categories = []
        self.n_categories = 0
        
    def setup(self, stage=None):
        """Setup data"""
        self.category_lines = {}
        self.all_categories = []
        
        # Read all name files
        for filename in glob.glob(os.path.join(self.data_path, '*.txt')):
            category = os.path.splitext(os.path.basename(filename))[0]
            self.all_categories.append(category)
            lines = readLines(filename)
            self.category_lines[category] = lines
        
        self.n_categories = len(self.all_categories)
        print(f"Numero de clases: {self.n_categories}")
    
    def train_dataloader(self):
        """Return a generator that yields random examples"""
        return DataLoader(self.category_lines, self.all_categories, num_samples=100000)
    
    def val_dataloader(self):
        """Return a generator for validation"""
        return DataLoader(self.category_lines, self.all_categories, num_samples=1000)
    
    def get_categories(self):
        """Get list of all categories"""
        return self.all_categories
    
    def get_n_categories(self):
        """Get number of categories"""
        return self.n_categories


class DataLoader:
    """Data loader with random sampling"""
    
    def __init__(self, category_lines, all_categories, num_samples=1000):
        self.category_lines = category_lines
        self.all_categories = all_categories
        self.num_samples = num_samples
        self.current_sample = 0
    
    def __iter__(self):
        self.current_sample = 0
        return self
    
    def __next__(self):
        if self.current_sample >= self.num_samples:
            raise StopIteration
        
        # Generate random example
        category, line, category_tensor, line_tensor = randomTrainingExample(
            self.category_lines, self.all_categories
        )
        
        self.current_sample += 1
        
        # Return in the format expected by Lightning (category_tensor first for consistency)
        return category_tensor[0], line_tensor  # Remove extra dimension from category_tensor
    
    def __len__(self):
        return self.num_samples