from ultralytics import YOLO
import sys

# Default path 
DEFAULT_WEIGHTS = "runs/detect/visdrone_final/weights/best.pt"

def export_model(weights_path):
    print(f"Loading weights from: {weights_path}")
    
    # Load the trained PyTorch model
    model = YOLO(weights_path)

    # Export to TensorRT
    # half=True enables FP16 (The speed boost)
    # simplify=True cleans up the ONNX graph before conversion
    print("Starting TensorRT Export (this may take 5-10 minutes)...")
    model.export(
        format='engine',
        device=0,
        half=True,
        simplify=True
    )
    
    print("Export Success!")

if __name__ == '__main__':
    # Allow running with a custom path arg: python src/export.py path/to/model.pt
    weights = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_WEIGHTS
    export_model(weights)