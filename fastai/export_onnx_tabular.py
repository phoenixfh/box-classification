#!/usr/bin/env python3
"""Export MLP tabular model to ONNX format."""

"""
PYTHONPATH=. python fastai/export_onnx_tabular.py \
    --model ./fastai_output/best.pt \
    --output ./fastai_output/best.onnx \
    --onnx_verify_data ./fastai/data.csv

"""    

import argparse
import sys
import warnings
from pathlib import Path
import torch
import torch.onnx
import onnxruntime as ort
import numpy as np

# Suppress deprecation warnings from torch.onnx.export
warnings.filterwarnings('ignore', category=DeprecationWarning, module='torch.onnx')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from models.mlp.network import MLP


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Export MLP model to ONNX',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--model', type=str, required=True,
                       help='PyTorch model checkpoint path (.pt file)')
    parser.add_argument('--output', type=str, default=None,
                       help='ONNX model output path (default: <model_name>.onnx)')
    parser.add_argument('--opset', type=int, default=18,
                       help='ONNX opset version')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for dummy input')
    parser.add_argument('--verify', action='store_true', default=True,
                       help='Verify exported model')
    parser.add_argument('--no_verify', dest='verify', action='store_false',
                       help='Skip verification')
    parser.add_argument('--onnx_verify_data', type=str, default=None,
                       help='CSV data file for ONNX verification (uses last 100 rows)')
    parser.add_argument('--target_col', type=str, default='label',
                       help='Target column name in verify data')
    
    return parser.parse_args()


def export_to_onnx(args):
    """Export PyTorch model to ONNX."""
    print(f"🔧 Exporting MLP model to ONNX...")
    print(f"   Model: {args.model}")
    
    # Load checkpoint
    print(f"   Loading checkpoint from {args.model}...")
    checkpoint = torch.load(args.model, map_location='cpu', weights_only=False)
    
    # Extract model config from checkpoint
    input_dim = checkpoint['input_dim']
    hidden_dims = checkpoint['hidden_dims']
    num_classes = checkpoint['num_classes']
    scaler = checkpoint['scaler']
    
    print(f"   Architecture: {input_dim} → {hidden_dims} → {num_classes}")
    
    # Create model
    print(f"   Creating model...")
    model = MLP(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=num_classes
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"   Model parameters: {model.get_num_params():,}")
    
    # Create dummy input
    dummy_input = torch.randn(args.batch_size, input_dim)
    print(f"   Dummy input shape: {dummy_input.shape}")
    
    # Determine output path
    if args.output is None:
        model_path = Path(args.model)
        args.output = str(model_path.parent / f"{model_path.stem}.onnx")
    
    print(f"📦 Exporting to ONNX...")
    print(f"   Output: {args.output}")
    print(f"   Opset version: {args.opset}")
    
    # Export to ONNX (use dynamo=False to avoid dynamic_axes warning)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        torch.onnx.export(
            model,
            dummy_input,
            args.output,
            input_names=['features'],
            output_names=['logits'],
            dynamic_axes={
                'features': {0: 'batch_size'},
                'logits': {0: 'batch_size'}
            },
            opset_version=args.opset,
            do_constant_folding=True,
            export_params=True,
            verbose=False,
            dynamo=False
        )
    
    print(f"✅ ONNX model saved: {args.output}")
    
    # Embed scaler parameters into ONNX model metadata
    print(f"📦 Embedding scaler parameters into ONNX model...")
    import json
    import onnx
    
    scaler_params = {
        'mean': scaler.mean_.tolist(),
        'scale': scaler.scale_.tolist(),
        'input_dim': input_dim,
        'num_classes': num_classes
    }
    
    # Load ONNX model and add metadata
    onnx_model = onnx.load(args.output)
    onnx_model.metadata_props.append(
        onnx.StringStringEntryProto(key='scaler_params', value=json.dumps(scaler_params))
    )
    onnx.save(onnx_model, args.output)
    print(f"✅ Scaler parameters embedded in ONNX model")
    
    # Verify model
    if args.verify:
        print(f"\n🔍 Verifying ONNX model...")
        verify_onnx_model(args.output, dummy_input.numpy(), model)
        
        # Verify with scaler
        print(f"\n🧪 Verifying with scaler (10 simulated rows)...")
        verify_with_scaler(args.output, scaler, input_dim, num_classes)
        
        # Verify with real data if provided
        if args.onnx_verify_data:
            print(f"\n📊 Verifying with real data: {args.onnx_verify_data}")
            verify_with_real_data(args.output, scaler, input_dim, args.onnx_verify_data, args.target_col)
    
    return args.output


def verify_onnx_model(onnx_path: str, test_input: np.ndarray, pytorch_model=None):
    """Verify ONNX model."""
    
    # Load ONNX model
    print(f"   Loading ONNX model...")
    session = ort.InferenceSession(onnx_path)
    
    # Get model info
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    print(f"   Input name: {input_name}")
    print(f"   Output name: {output_name}")
    
    # Run inference
    print(f"   Running ONNX inference...")
    onnx_outputs = session.run(
        [output_name],
        {input_name: test_input.astype(np.float32)}
    )
    
    onnx_logits = onnx_outputs[0]
    print(f"   ONNX output shape: {onnx_logits.shape}")
    print(f"   ONNX predicted class: {np.argmax(onnx_logits, axis=1)}")
    
    # Compare with PyTorch if model is provided
    if pytorch_model is not None:
        print(f"   Comparing with PyTorch output...")
        pytorch_model.eval()
        with torch.no_grad():
            pytorch_input = torch.tensor(test_input, dtype=torch.float32)
            pytorch_logits = pytorch_model(pytorch_input).numpy()
        
        # Compute difference
        max_diff = np.abs(pytorch_logits - onnx_logits).max()
        print(f"   PyTorch output shape: {pytorch_logits.shape}")
        print(f"   PyTorch predicted class: {np.argmax(pytorch_logits, axis=1)}")
        print(f"   Max difference: {max_diff:.6f}")
        
        if max_diff < 1e-5:
            print(f"   ✅ Outputs match!")
        else:
            print(f"   ⚠️  Outputs differ by {max_diff:.6f}")
    
    print(f"✅ ONNX model verification complete!")


def verify_with_scaler(onnx_path: str, scaler, input_dim: int, num_classes: int):
    """Verify ONNX model with scaler normalization using simulated data."""
    
    # Generate 10 rows of simulated data (before normalization)
    print(f"   Generating 10 rows of simulated data...")
    np.random.seed(42)
    raw_data = np.random.randn(10, input_dim).astype(np.float32) * 10  # Scale up for realistic range
    
    print(f"   Raw data shape: {raw_data.shape}")
    print(f"   Raw data sample (first row): {raw_data[0, :5]}...")
    
    # Normalize using training scaler
    print(f"   Normalizing with training scaler...")
    normalized_data = scaler.transform(raw_data).astype(np.float32)
    
    print(f"   Normalized data sample (first row): {normalized_data[0, :5]}...")
    
    # Load ONNX model
    print(f"   Loading ONNX model...")
    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    # Run inference
    print(f"   Running ONNX inference...")
    outputs = session.run([output_name], {input_name: normalized_data})
    logits = outputs[0]
    predictions = np.argmax(logits, axis=1)
    
    # Print results
    print(f"\n📊 Inference Results (10 rows):")
    print(f"   Logits shape: {logits.shape}")
    print(f"   Predictions: {predictions}")
    
    for i in range(10):
        # stable softmax
        logits_i = logits[i] - np.max(logits[i])
        probs = np.exp(logits_i) / np.sum(np.exp(logits_i))
        print(f"   Row {i}: predicted={predictions[i]}, probs={probs}")
    
    print(f"\n✅ Scaler verification complete!")


def verify_with_real_data(onnx_path: str, scaler, input_dim: int, data_path: str, target_col: str):
    """Verify ONNX model with real data from CSV file (last 100 rows)."""
    import pandas as pd
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    
    # Load data
    print(f"   Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Get last 100 rows
    test_df = df.tail(100)
    print(f"   Testing on last {len(test_df)} rows...")
    
    # Get feature columns (all columns except target)
    feature_cols = [c for c in df.columns if c != target_col]
    print(f"   Feature columns: {len(feature_cols)}")
    
    # Extract features and labels
    X_test = test_df[feature_cols].values.astype(np.float32)
    y_test = test_df[target_col].values.astype(np.int64)
    
    # Adjust labels if needed
    y_min = y_test.min()
    if y_min != 0:
        y_test = y_test - y_min
    
    # Normalize using training scaler
    print(f"   Normalizing with training scaler...")
    X_test_scaled = scaler.transform(X_test).astype(np.float32)
    
    # Load ONNX model
    print(f"   Loading ONNX model...")
    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    # Run inference
    print(f"   Running ONNX inference...")
    outputs = session.run([output_name], {input_name: X_test_scaled})
    logits = outputs[0]
    predictions = np.argmax(logits, axis=1)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)
    
    # Print results
    print(f"\n📊 ONNX Inference Results (last 100 rows):")
    print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"\n   Classification Report:")
    print(classification_report(y_test, predictions))
    print(f"\n   Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    
    # Show sample predictions
    print(f"\n   Sample Predictions (first 10):")
    for i in range(min(10, len(predictions))):
        status = "✓" if y_test[i] == predictions[i] else "✗"
        print(f"      Row {len(df)-100+i}: True={y_test[i]}, Predicted={predictions[i]} {status}")
    
    print(f"\n✅ Real data verification complete!")


def benchmark_onnx(onnx_path: str, input_dim: int, num_runs: int = 100):
    """Benchmark ONNX model inference."""
    import time
    
    print(f"\n⚡ Benchmarking ONNX inference...")
    
    # Load model
    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    # Prepare input
    test_input = np.random.randn(1, input_dim).astype(np.float32)
    
    # Warmup
    for _ in range(10):
        session.run([output_name], {input_name: test_input})
    
    # Benchmark
    start_time = time.time()
    for _ in range(num_runs):
        session.run([output_name], {input_name: test_input})
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs * 1000  # ms
    throughput = num_runs / (end_time - start_time)
    
    print(f"   Runs: {num_runs}")
    print(f"   Average latency: {avg_time:.3f} ms")
    print(f"   Throughput: {throughput:.1f} samples/s")


if __name__ == '__main__':
    args = parse_args()
    onnx_path = export_to_onnx(args)
