"""MLP network architecture for tabular data classification."""

import torch
import torch.nn as nn
from typing import List
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) for tabular data classification.
    
    Args:
        input_dim: Input feature dimension (e.g., 200)
        hidden_dims: List of hidden layer dimensions (e.g., [512, 256, 128])
        output_dim: Output dimension (number of classes, e.g., 4)
        dropout: Dropout probability
        activation: Activation function ('relu', 'leaky_relu', 'elu')
        batch_norm: Whether to use Batch Normalization
    
    Example:
        >>> model = MLP(input_dim=200, hidden_dims=[512, 256, 128], output_dim=4)
        >>> x = torch.randn(32, 200)
        >>> y = model(x)  # Output: [32, 4]
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout: float = 0.5,
        activation: str = 'relu',
        batch_norm: bool = True
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        
        for i in range(len(dims) - 1):
            # Linear layer
            layers.append(nn.Linear(dims[i], dims[i+1]))
            
            # Skip activation and normalization for output layer
            if i < len(dims) - 2:
                # Batch Normalization
                if batch_norm:
                    layers.append(nn.BatchNorm1d(dims[i+1]))
                
                # Activation function
                if activation == 'relu':
                    layers.append(nn.ReLU())
                elif activation == 'leaky_relu':
                    layers.append(nn.LeakyReLU(0.2))
                elif activation == 'elu':
                    layers.append(nn.ELU())
                elif activation == 'gelu':
                    layers.append(nn.GELU())
                else:
                    raise ValueError(f"Unknown activation: {activation}")
                
                # Dropout
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
        
        Returns:
            Output logits of shape [batch_size, output_dim]
        """
        return self.network(x)
    
    def get_num_params(self):
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def __repr__(self):
        num_params = self.get_num_params()
        return (
            f"MLP(input_dim={self.input_dim}, "
            f"hidden_dims={self.hidden_dims}, "
            f"output_dim={self.output_dim}, "
            f"num_params={num_params:,})"
        )


class TabularDataset(Dataset):
    """
    PyTorch Dataset for tabular data.
    
    Args:
        csv_path: Path to CSV file
        feature_cols: List of feature column names
        target_col: Target column name
        normalize: Whether to normalize features
        scaler: Pre-fitted scaler (for test set)
    
    Example:
        >>> dataset = TabularDataset(
        ...     csv_path='data.csv',
        ...     feature_cols=['feature_0', 'feature_1', ..., 'feature_199'],
        ...     target_col='label'
        ... )
        >>> x, y = dataset[0]
    """
    
    def __init__(
        self,
        csv_path: str = None,
        feature_cols: List[str] = None,
        target_col: str = None,
        normalize: bool = True,
        scaler: StandardScaler = None,
        X: np.ndarray = None,
        y: np.ndarray = None
    ):
        if csv_path is not None:
            # Load from CSV
            self.df = pd.read_csv(csv_path)
            self.feature_cols = feature_cols
            self.target_col = target_col
            
            # Extract features and labels
            self.X = self.df[feature_cols].values.astype(np.float32)
            self.y = self.df[target_col].values.astype(np.int64)
        else:
            # Use provided arrays
            self.X = X.astype(np.float32)
            self.y = y.astype(np.int64)
        
        # Normalize features
        self.scaler = scaler
        if normalize:
            if self.scaler is None:
                self.scaler = StandardScaler()
                self.X = self.scaler.fit_transform(self.X)
            else:
                self.X = self.scaler.transform(self.X)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        """Get a single sample."""
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])
    
    def get_scaler(self):
        """Get the fitted scaler."""
        return self.scaler
    
    def get_num_classes(self):
        """Get number of unique classes."""
        return len(np.unique(self.y))
    
    def get_class_distribution(self):
        """Get class distribution."""
        unique, counts = np.unique(self.y, return_counts=True)
        return dict(zip(unique, counts))
