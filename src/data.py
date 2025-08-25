import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Any
from omegaconf import DictConfig
from .utils import is_square
import hydra
import os

class DataLoader:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

        # Get the original working directory (before Hydra changed it)
        self.original_cwd = hydra.utils.get_original_cwd()
        
        # Use configured data directory or default to 'data'
        data_dir_name = cfg.get('paths', {}).get('data_dir', 'data')
        self.data_dir = os.path.join(self.original_cwd, data_dir_name)
        
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load dataset based on configuration"""
        if self.cfg.dataset.name == "cifar10":
            return self._load_cifar10()
        elif self.cfg.dataset.name == "mnist":
            return self._load_mnist()
        elif self.cfg.dataset.name == "synthetic":
            return self._load_synthetic()
        else:
            raise ValueError(f"Unknown dataset: {self.cfg.dataset.name}")
    
    def _load_cifar10(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load CIFAR-10 dataset"""
        transform = transforms.Compose([transforms.ToTensor()])
        
        trainset = torchvision.datasets.CIFAR10(
            root=self.data_dir, train=True, download=True, transform=transform
        )
        testset = torchvision.datasets.CIFAR10(
            root=self.data_dir, train=False, download=True, transform=transform
        )
        
        # Convert to numpy arrays
        X_train = np.array([np.array(x) for x, _ in trainset])
        y_train = np.array([y for _, y in trainset])
        X_test = np.array([np.array(x) for x, _ in testset])
        y_test = np.array([y for _, y in testset])
        
        # Apply preprocessing
        X_train, y_train, X_test, y_test = self._preprocess(X_train, y_train, X_test, y_test)
        
        return X_train, y_train, X_test, y_test
    
    def _load_mnist(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load MNIST dataset"""
        transform = transforms.Compose([transforms.ToTensor()])
        
        trainset = torchvision.datasets.MNIST(
            root=self.data_dir, train=True, download=True, transform=transform
        )
        testset = torchvision.datasets.MNIST(
            root=self.data_dir, train=False, download=True, transform=transform
        )
        
        X_train = np.array([np.array(x) for x, _ in trainset])
        y_train = np.array([y for _, y in trainset])
        X_test = np.array([np.array(x) for x, _ in testset])
        y_test = np.array([y for _, y in testset])
        
        X_train, y_train, X_test, y_test = self._preprocess(X_train, y_train, X_test, y_test)
        
        return X_train, y_train, X_test, y_test
    
    def _load_synthetic(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate synthetic Gaussian dataset"""
        np.random.seed(self.cfg.dataset.random_state)
        assert is_square(self.cfg.dataset.n_features_squared), "n_features_squared must be a square number"

        X = np.random.normal(0.0, self.cfg.dataset.variance , (self.cfg.dataset.n_samples, self.cfg.dataset.n_features_squared))
        y = np.random.randint(0, self.cfg.dataset.n_classes, self.cfg.dataset.n_samples)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.cfg.training.test_split, 
            random_state=self.cfg.seed
        )
        
        X_train, y_train, X_test, y_test = self._preprocess(X_train, y_train, X_test, y_test)
        
        return X_train, y_train, X_test, y_test
    
    def _preprocess(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Apply preprocessing based on configuration"""
        
        # Flatten if needed
        if self.cfg.dataset.get('flatten', False):
            X_train = X_train.reshape(X_train.shape[0], -1)
            X_test = X_test.reshape(X_test.shape[0], -1)
        
        # Normalize to [0, 1]
        if self.cfg.dataset.get('normalize', False):
            X_train = X_train.astype(np.float32) / 255.0
            X_test = X_test.astype(np.float32) / 255.0
        
        # Center data (make mean = 0)
        if self.cfg.dataset.get('center_data', False) or self.cfg.model.get('center_data', False):
            mean = np.mean(X_train, axis=0)
            X_train = X_train - mean
            X_test = X_test - mean
        
        # Apply data poison
        X_train, y_train = self._apply_poison(X_train, y_train)
        X_test, y_test = self._apply_poison(X_test, y_test)
        
        return X_train, y_train, X_test, y_test
    

    def _apply_poison(self, X: np.ndarray, y: np.ndarray):
        """Apply data poison for experiments (vectorized, L2-normalized poison vector)"""
        if not hasattr(self.cfg, 'poison'):
            return X, y
        
        proportion = self.cfg.poison.get('proportion', 0)
        if proportion <= 0:
            return X, y
        

        # Number of samples to poison
        n_poison = int(len(X) * proportion)

        # Get feature shape and size
        feature_shape = X.shape[1:]
        feature_size = np.prod(feature_shape)

        # Poison mask: last 4 pixels = 1
        pois_flat = np.zeros(feature_size, dtype=np.float32)
        pois_flat[-self.cfg.poison['num_pixels']:] = 1.0

        # Normalize to L2 norm = 5
        norm = np.linalg.norm(pois_flat, ord=2)
        if norm > 0:
            pois_flat = pois_flat * (self.cfg.poison['L2_size']/ norm)

        # Reshape back
        pois = pois_flat.reshape(feature_shape)

        # Copy so original dataset isn’t modified
        X_poisoned = X.copy()
        y_poisoned = y.copy()

        # Find indices where y != 0
        nonzero_indices = np.where(y != 0)[0]

        # Take only the first n_poison of those
        mask = nonzero_indices[:n_poison]

        # Apply poison
        X_poisoned[mask] += pois
        y_poisoned[mask] = self.cfg.poison['into_class']
    
        return X_poisoned, y