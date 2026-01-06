#!/usr/bin/env python3
"""
YOLO Model Export Script
Export YOLO .pt model to ONNX format with dynamic batch size and fixed image size
"""

import argparse
from pathlib import Path
from ultralytics import YOLO
import onnx
from onnx import shape_inference


def export_yolo_to_onnx(model_path: str, imgsz: int = 640, output_path: str = None):
    """
    Export YOLO model to ONNX format with dynamic batch size
    
    Args:
        model_path: Path to YOLO .pt model file
        imgsz: Fixed image size (default: 640)
        output_path: Output ONNX file path (default: same as model_path with .onnx extension)
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"Loading YOLO model from: {model_path}")
    model = YOLO(str(model_path))
    
    if output_path is None:
        output_path = model_path.with_suffix('.onnx')
    
    print(f"Exporting to ONNX format...")
    print(f"  Image size: {imgsz}")
    print(f"  Batch size: dynamic")
    print(f"  Output path: {output_path}")
    
    # Export with full dynamic dimensions first
    model.export(
        format='onnx',
        imgsz=imgsz,
        dynamic=True,  # Export with dynamic batch and image size
        simplify=True
    )
    
    # Modify ONNX model to fix image size while keeping batch dynamic
    onnx_path = model_path.with_suffix('.onnx') if output_path is None else Path(output_path)
    print(f"\nModifying ONNX model to fix image size to {imgsz}x{imgsz}...")
    
    onnx_model = onnx.load(str(onnx_path))
    
    # Fix image dimensions (height and width) for inputs
    for input_tensor in onnx_model.graph.input:
        shape = input_tensor.type.tensor_type.shape
        # Keep batch dynamic, keep channels, fix height and width
        if len(shape.dim) == 4:  # NCHW format
            shape.dim[2].dim_value = imgsz  # height
            shape.dim[2].ClearField('dim_param')
            shape.dim[3].dim_value = imgsz  # width
            shape.dim[3].ClearField('dim_param')
    
    # Save modified model
    onnx.save(onnx_model, str(onnx_path))
    
    print(f"Modified ONNX model saved with dynamic batch size and fixed image size {imgsz}x{imgsz}")
    
    print(f"\nExport completed successfully!")
    print(f"ONNX model saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Export YOLO model to ONNX with dynamic batch size'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        help='Path to YOLO .pt model file'
    )
    parser.add_argument(
        '--imgsz',
        type=int,
        default=640,
        help='Fixed image size (default: 640)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output ONNX file path (default: same as input with .onnx extension)'
    )
    
    args = parser.parse_args()
    
    export_yolo_to_onnx(
        model_path=args.model_path,
        imgsz=args.imgsz,
        output_path=args.output
    )


if __name__ == '__main__':
    main()
