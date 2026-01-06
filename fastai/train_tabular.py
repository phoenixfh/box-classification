#!/usr/bin/env python3
"""FastAI Tabular Data Training - MLP Multi-Layer Perceptron"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import mlflow
import mlflow.pytorch
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))  # Add project root for utils
sys.path.insert(0, str(Path(__file__).parent))         # Add fastai dir for models

from models.mlp.network import MLP

try:
    from utils import (
        is_main_process, setup_mlflow,
    )
except ImportError:
    # Fallback if utils not found - provide dummy functions
    def is_main_process():
        return True
    def setup_mlflow(project_name, task_name):
        import mlflow
        mlflow.set_experiment(project_name)
        return mlflow.start_run(run_name=task_name)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='FastAI Tabular Data Training - MLP',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument('--data', type=str, required=True,
                       help='Path to CSV data file')
    parser.add_argument('--feature_cols', type=str, default=None,
                       help='Comma-separated feature column names')
    parser.add_argument('--feature_prefix', type=str, default=None,
                       help='Feature column prefix (e.g., "feature_")')
    parser.add_argument('--target_col', type=str, default='label',
                       help='Target column name')
    parser.add_argument('--use_all_features', action='store_true',
                       help='Use all columns except target as features')
    
    # Model arguments
    parser.add_argument('--hidden_dims', type=str, default='512,256,128',
                       help='Comma-separated hidden layer dimensions')
    parser.add_argument('--dropout', type=float, default=0.5,
                       help='Dropout probability')
    parser.add_argument('--activation', type=str, default='relu',
                       choices=['relu', 'leaky_relu', 'elu', 'gelu'],
                       help='Activation function')
    parser.add_argument('--batch_norm', action='store_true', default=True,
                       help='Use batch normalization')
    parser.add_argument('--no_batch_norm', dest='batch_norm', action='store_false',
                       help='Disable batch normalization')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                       help='Weight decay (L2 regularization)')
    parser.add_argument('--val_split', type=float, default=0.2,
                       help='Validation split ratio')
    parser.add_argument('--early_stopping', type=int, default=20,
                       help='Early stopping patience (0 to disable)')
    
    # MLflow arguments
    parser.add_argument('--mlflow_uri', type=str,
                       default='http://192.168.16.130:5000/',
                       help='MLflow tracking URI')
    parser.add_argument('--project_name', type=str, default='mlp-classification',
                       help='MLflow experiment/project name')
    parser.add_argument('--task_name', type=str, default='mlp-exp',
                       help='MLflow run name')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, 
                       help='Output directory for models and logs')
    parser.add_argument('--save_best', action='store_true', default=True,
                       help='Save best model')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use')
   
   
    args_t = parser.parse_args()

    if args_t.output_dir is None:
        args_t.output_dir = 'runs/' + args_t.project_name + '/' + args_t.task_name
        
    args_t.output_model = args_t.output_dir + '/best.pt'
    return args_t


def set_seed(seed):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_feature_columns(df, args):
    """Determine feature columns based on arguments."""
    if args.feature_cols is not None:
        # Explicitly specified columns
        feature_cols = args.feature_cols.split(',')
        feature_cols = [c.strip() for c in feature_cols]
    elif args.feature_prefix is not None:
        # Match columns by prefix
        feature_cols = [c for c in df.columns if c.startswith(args.feature_prefix)]
    elif args.use_all_features:
        # Use all columns except target
        feature_cols = [c for c in df.columns if c != args.target_col]
    else:
        raise ValueError(
            "Must specify one of: --feature_cols, --feature_prefix, or --use_all_features"
        )
    
    # Validate columns exist
    missing = set(feature_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Feature columns not found in CSV: {missing}")
    
    if args.target_col not in df.columns:
        raise ValueError(f"Target column not found in CSV: {args.target_col}")
    
    return feature_cols


def load_data(args):
    """Load and preprocess data."""
    print(f"📁 Loading data from: {args.data}")
    
    # Load train CSV
    df_train = pd.read_csv(args.data)
    print(f"   Train samples: {len(df_train)}")
    
    # Load validation CSV by replacing /train/ with /val/
    val_data_path = args.data.replace('/train/', '/val/')
    print(f"📁 Loading validation data from: {val_data_path}")
    df_val = pd.read_csv(val_data_path)
    print(f"   Val samples: {len(df_val)}")
    
    # Get feature columns from train set
    feature_cols = get_feature_columns(df_train, args)
    print(f"   Feature columns: {len(feature_cols)}")
    print(f"   First 5 features: {feature_cols[:5]}")
    if len(feature_cols) > 5:
        print(f"   ... ({len(feature_cols) - 5} more)")
    
    # Extract features and labels for train set
    X_train = df_train[feature_cols].values.astype(np.float64)
    y_train = df_train[args.target_col].values.astype(np.int64)
    
    # Extract features and labels for validation set
    X_val = df_val[feature_cols].values.astype(np.float64)
    y_val = df_val[args.target_col].values.astype(np.int64)
    
    # Clean train data: replace inf and very large values
    print(f"   Cleaning train data (removing inf/nan values)...")
    X_train[np.isinf(X_train)] = np.nan
    float32_max = np.finfo(np.float32).max
    X_train[np.abs(X_train) > float32_max] = np.nan
    
    if np.isnan(X_train).any():
        nan_count = np.isnan(X_train).sum()
        print(f"   ⚠️  Found {nan_count} NaN/Inf values in train data, filling with column means...")
        col_means = np.nanmean(X_train, axis=0)
        for i in range(X_train.shape[1]):
            col_mask = np.isnan(X_train[:, i])
            if col_mask.any():
                X_train[col_mask, i] = col_means[i] if not np.isnan(col_means[i]) else 0.0
    
    X_train = X_train.astype(np.float32)
    
    # Clean validation data: replace inf and very large values
    print(f"   Cleaning validation data (removing inf/nan values)...")
    X_val[np.isinf(X_val)] = np.nan
    X_val[np.abs(X_val) > float32_max] = np.nan
    
    if np.isnan(X_val).any():
        nan_count = np.isnan(X_val).sum()
        print(f"   ⚠️  Found {nan_count} NaN/Inf values in validation data, filling with column means...")
        col_means_val = np.nanmean(X_val, axis=0)
        for i in range(X_val.shape[1]):
            col_mask = np.isnan(X_val[:, i])
            if col_mask.any():
                X_val[col_mask, i] = col_means_val[i] if not np.isnan(col_means_val[i]) else 0.0
    
    X_val = X_val.astype(np.float32)
    
    if np.isnan(y_train).any() or np.isnan(y_val).any():
        raise ValueError("Target contains NaN values. Please clean your data.")
    
    # Ensure labels start from 0
    y_train_min = y_train.min()
    if y_train_min != 0:
        print(f"   ⚠️  Train labels start from {y_train_min}, shifting to start from 0")
        y_train = y_train - y_train_min
    
    y_val_min = y_val.min()
    if y_val_min != 0:
        print(f"   ⚠️  Val labels start from {y_val_min}, shifting to start from 0")
        y_val = y_val - y_val_min
    
    # Get class info from train set
    num_classes = len(np.unique(y_train))
    unique_classes = np.unique(y_train)
    print(f"   Classes: {num_classes} ({list(unique_classes)})")
    
    # Class distribution for train
    unique, counts = np.unique(y_train, return_counts=True)
    print(f"   Train class distribution:")
    for cls, cnt in zip(unique, counts):
        print(f"      Class {cls}: {cnt} samples ({cnt/len(y_train)*100:.1f}%)")
    
    # Class distribution for val
    unique_val, counts_val = np.unique(y_val, return_counts=True)
    print(f"   Val class distribution:")
    for cls, cnt in zip(unique_val, counts_val):
        print(f"      Class {cls}: {cnt} samples ({cnt/len(y_val)*100:.1f}%)")
    
    # Normalize features
    print(f"   Normalizing features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    
    return X_train, X_val, y_train, y_val, num_classes, len(feature_cols), scaler


def create_dataloaders(X_train, X_val, y_train, y_val, batch_size):
    """Create PyTorch DataLoaders."""
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long)
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    
    return train_loader, val_loader


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training', leave=False)
    for batch_X, batch_y in pbar:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += batch_y.size(0)
        correct += predicted.eq(batch_y).sum().item()
        
        pbar.set_postfix({'loss': loss.item(), 'acc': 100.*correct/total})
    
    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def validate(model, val_loader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total
    
    return avg_loss, accuracy, np.array(all_preds), np.array(all_labels)


def train_mlp(args):
    """Main training function."""
    # Set seed
    set_seed(args.seed)
    
    # Setup device - Auto-detect GPU/CPU
    print(f"\n🖥️  GPU Detection:")
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   CUDA device count: {torch.cuda.device_count()}")
        print(f"   CUDA device name: {torch.cuda.get_device_name(0)}")
    else:
        print(f"   ⚠️  WARNING: PyTorch cannot detect CUDA!")
        print(f"   This may be because:")
        print(f"      1. PyTorch was installed without CUDA support (CPU-only)")
        print(f"      2. CUDA libraries are not in LD_LIBRARY_PATH")
        print(f"   To fix, install PyTorch with CUDA support:")
        print(f"      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"\n   Selected device: {device}")
    
    # Load data
    X_train, X_val, y_train, y_val, num_classes, input_dim, scaler = load_data(args)
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        X_train, X_val, y_train, y_val, args.batch_size
    )
    
    # Parse hidden dimensions
    hidden_dims = [int(x.strip()) for x in args.hidden_dims.split(',')]
    
    # Create model
    print(f"\n🏗️  Creating model...")
    model = MLP(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=num_classes,
        dropout=args.dropout,
        activation=args.activation,
        batch_norm=args.batch_norm
    )
    model = model.to(device)
    print(f"   {model}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    # Setup MLflow with timeout protection
    print(f"\n📊 Setting up MLflow...")
    use_mlflow = True
    try:
        import socket
        socket.setdefaulttimeout(5)

        # Configure AWS/MinIO credentials for MLflow artifact storage (uppercase keys required)
        # Note: These credentials must match your MinIO server configuration
        # If you get "SignatureDoesNotMatch" errors, verify:
        # 1. ACCESS_KEY and SECRET_KEY match MinIO server
        # 2. Endpoint URL is correct (http vs https)
        # 3. Region is set (use 'us-east-1' for MinIO)

        # 设置MinIO访问凭据
        os.environ['AWS_ACCESS_KEY_ID'] = 'mlflow'
        os.environ['AWS_SECRET_ACCESS_KEY'] = 'mlflow@SN'
        os.environ['AWS_ENDPOINT_URL'] = 'http://192.168.16.130:9000'
        os.environ['AWS_REGION'] = ''
        os.environ['MLFLOW_S3_IGNORE_TLS'] = 'true'

        mlflow.set_tracking_uri(args.mlflow_uri)
        mlflow.set_experiment(args.project_name)



        # mlflow_run = setup_mlflow(args.project_name, args.task_name)
    except Exception as e:
        print(f"   ⚠️  MLflow connection failed: {e}")
        print(f"   Continuing without MLflow logging...")
        use_mlflow = False
    
    # Training function wrapper
    def run_training():
        # Training loop
        print(f"\n🚀 Starting training for {args.epochs} epochs...")
        best_val_acc = 0.0
        patience_counter = 0
        
        # Log parameters if using MLflow
        if use_mlflow:
            params = {
                'input_dim': input_dim,
                'hidden_dims': str(hidden_dims),
                'output_dim': num_classes,
                'dropout': args.dropout,
                'activation': args.activation,
                'batch_norm': args.batch_norm,
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'lr': args.lr,
                'weight_decay': args.weight_decay,
                'optimizer': 'adam',
                'num_params': model.get_num_params(),
            }
            mlflow.log_params(params)
        
        for epoch in range(args.epochs):
            # Train
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, device
            )
            
            # Validate
            val_loss, val_acc, val_preds, val_labels = validate(
                model, val_loader, criterion, device
            )
            
            # Update learning rate
            scheduler.step(val_acc)
            
            # Log metrics if using MLflow
            if use_mlflow:
                metrics = {
                    'train/loss': train_loss,
                    'train/acc': train_acc,
                    'val/loss': val_loss,
                    'val/acc': val_acc,
                    'lr': optimizer.param_groups[0]['lr'],
                }
                mlflow.log_metrics(metrics, step=epoch)
            
            # Print progress
            print(f"Epoch {epoch+1}/{args.epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                
                if args.save_best:
                    # Ensure output directory exists
                    os.makedirs(args.output_dir, exist_ok=True)
                    
                    # Save to output_dir (for MLflow artifact logging)
                    model_path = os.path.join(args.output_dir, 'best.pt')
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'scaler': scaler,
                        'num_classes': num_classes,
                        'input_dim': input_dim,
                        'hidden_dims': hidden_dims,
                    }, model_path)
                    print(f"   💾 Saved best model to {model_path} (acc: {best_val_acc:.4f})")

            else:
                patience_counter += 1
            
            # Early stopping
            if args.early_stopping > 0 and patience_counter >= args.early_stopping:
                print(f"⏹️  Early stopping triggered (patience: {args.early_stopping})")
                break
        
        # Final evaluation
        print(f"\n📈 Training completed!")
        print(f"   Best validation accuracy: {best_val_acc:.4f}")
        
        # Load best model for final evaluation
        if args.save_best and os.path.exists(model_path):
            print(f"   Loading best model from {model_path} for final evaluation...")
            checkpoint = torch.load(model_path, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
        
        # Final validation
        _, final_acc, final_preds, final_labels = validate(
            model, val_loader, criterion, device
        )
        
        # Classification report
        print(f"\n📊 Classification Report:")
        print(classification_report(final_labels, final_preds))
        
        # Log final metrics if using MLflow
        if use_mlflow:
            mlflow.log_metrics({
                'final/best_val_acc': best_val_acc,
                'final/val_acc': final_acc,
            })
            
            # Log artifacts
            if args.save_best:
                try:
                    # Ensure model file exists before logging
                    if os.path.exists(model_path):
                        print(f"   📤 Logging model artifact to MLflow: {model_path}")
                        mlflow.log_artifact(model_path, artifact_path='models')
                        print(f"   ✅ Model artifact logged successfully")
                    else:
                        print(f"   ⚠️  Model file not found: {model_path}")
                    
                        
                except Exception as e:
                    print(f"   ⚠️  Could not log artifact to MLflow: {e}")
                    print(f"   Model saved locally at: {model_path}")
                    import traceback
                    traceback.print_exc()
        
        print(f"\n✅ Training complete! Best accuracy: {best_val_acc:.4f}")
        return best_val_acc
    
    # Run training with or without MLflow
    if use_mlflow:
        with mlflow.start_run(run_name=args.task_name) as run:
            print(f"   Run ID: {run.info.run_id}")
            best_val_acc = run_training()
    else:
        best_val_acc = run_training()
    
    return model, X_train, X_val, y_train, y_val, device


def test_inference(model_path, data_path, args):
    """Test inference on last 100 rows of data."""
    print(f"\n🧪 Testing inference with {model_path}...")
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    # Load validation CSV by replacing /train/ with /val/
    val_data_path = data_path.replace('/train/', '/val/')
    print(f"   Loading validation data from: {val_data_path}")
    
    # Load data
    df = pd.read_csv(val_data_path)
    feature_cols = get_feature_columns(df, args)
    
    # Get last 100 rows
    test_df = df.tail(100)
    print(f"   Testing on last {len(test_df)} rows...")
    
    X_test = test_df[feature_cols].values.astype(np.float64)
    y_test = test_df[args.target_col].values.astype(np.int64)
    
    # Clean test data: replace inf and very large values
    print(f"   Cleaning test data (removing inf/nan values)...")
    X_test[np.isinf(X_test)] = np.nan
    float32_max = np.finfo(np.float32).max
    X_test[np.abs(X_test) > float32_max] = np.nan
    
    if np.isnan(X_test).any():
        nan_count = np.isnan(X_test).sum()
        print(f"   ⚠️  Found {nan_count} NaN/Inf values in test data, filling with column means...")
        col_means = np.nanmean(X_test, axis=0)
        for i in range(X_test.shape[1]):
            col_mask = np.isnan(X_test[:, i])
            if col_mask.any():
                X_test[col_mask, i] = col_means[i] if not np.isnan(col_means[i]) else 0.0
    
    X_test = X_test.astype(np.float32)
    
    # Adjust labels if needed
    y_min = y_test.min()
    if y_min != 0:
        y_test = y_test - y_min
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    scaler = checkpoint['scaler']
    num_classes = checkpoint['num_classes']
    input_dim = checkpoint['input_dim']
    hidden_dims = checkpoint['hidden_dims']
    
    # Normalize using saved scaler
    X_test_scaled = scaler.transform(X_test)
    
    # Load model
    model = MLP(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=num_classes,
        dropout=args.dropout,
        activation=args.activation,
        batch_norm=args.batch_norm
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Inference
    with torch.no_grad():
        X_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
        outputs = model(X_tensor)
        _, predictions = outputs.max(1)
        predictions = predictions.cpu().numpy()
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)
    
    print(f"\n📊 Inference Results:")
    print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"\n   Classification Report:")
    print(classification_report(y_test, predictions))
    print(f"\n   Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    
    # Show sample predictions
    print(f"\n   Sample Predictions (first 10):")
    for i in range(min(10, len(predictions))):
        print(f"      Row {len(df)-100+i}: True={y_test[i]}, Predicted={predictions[i]}")
    
    print(f"\n✅ Inference test complete!")


if __name__ == '__main__':
    args = parse_args()


    model, X_train, X_val, y_train, y_val, device = train_mlp(args)

    print("all args:", args)

    print("args.output_model:", args.output_model)

    
    # Test inference on last 100 rows
    if os.path.exists(args.output_model):
        print("\n🧪 Testing inference on the saved model...")
        test_inference(args.output_model, args.data, args)
        
        # Automatically export to ONNX
        print("\n🔄 Automatically exporting model to ONNX...")
        try:
            import subprocess
            onnx_output_path = args.output_model.replace('.pt', '.onnx')
            
            # Get validation data path
            val_data_path = args.data.replace('/train/', '/val/')
            
            export_cmd = [
                sys.executable,
                'fastai/export_onnx_tabular.py',
                '--model', args.output_model,
                '--output', onnx_output_path,
                '--onnx_verify_data', val_data_path,
                '--target_col', args.target_col
            ]
            
            print(f"   Running: {' '.join(export_cmd)}")
            result = subprocess.run(export_cmd, check=True, capture_output=False, text=True)
            
            print(f"\n✅ ONNX model exported successfully to: {onnx_output_path}")
            
        except subprocess.CalledProcessError as e:
            print(f"\n⚠️  ONNX export failed: {e}")
            print(f"   You can manually export using:")
            print(f"   python fastai/export_onnx_tabular.py --model {args.output_model} --output {onnx_output_path}")
        except Exception as e:
            print(f"\n⚠️  ONNX export failed: {e}")
            print(f"   You can manually export using:")
            print(f"   python fastai/export_onnx_tabular.py --model {args.output_model}")
