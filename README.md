# Drone Detection with TensorRT Optimization

A high-performance object detection system for drone surveillance using YOLOv11 and TensorRT optimization. This project trains a YOLO model on the VisDrone dataset and exports it to TensorRT for real-time inference on edge devices.

## Project Overview

This project implements an end-to-end pipeline for detecting objects from drone imagery:
- **Dataset**: VisDrone2019 Detection dataset (converted from VisDrone to YOLO format)
- **Model**: YOLOv11s (small variant for balanced speed/accuracy)
- **Optimization**: TensorRT FP16 for GPU acceleration
- **Target Hardware**: NVIDIA RTX 4050 (optimized for edge deployment)

### Key Features
- Automated dataset conversion from VisDrone to YOLO format
- Class consolidation (10 classes → 3: Person, Vehicle, Cycle)
- YOLOv11 training with GPU acceleration
- TensorRT FP16 export for production deployment
- Real-time video inference with performance metrics
- Comprehensive benchmarking notebooks

## Performance Results

Our benchmarks show significant performance improvements with TensorRT optimization:

| Metric | PyTorch | TensorRT | Improvement |
|--------|---------|----------|-------------|
| **Inference Speed (FPS)** | ~45 FPS | ~110 FPS | **2.4x faster** |
| **Inference Latency** | ~22 ms | ~9 ms | **59% reduction** |
| **mAP@50** | ~0.45 | ~0.45 | Maintained |
| **Model Size** | 19.2 MB | 22.0 MB | +14.6% (includes optimized kernels) |

> **Note**: TensorRT engines are larger because they include pre-compiled, hardware-specific optimized kernels for your GPU. This trade-off delivers significantly faster inference speeds.

### Why TensorRT is Larger but Better:
- Pre-optimized GPU kernels specific to RTX 4050
- Fused operations and optimized computation graph
- FP16 precision constants and specialized data structures
- **Result**: 2.4x speed improvement for minimal storage cost

## Project Structure

```
drone_detection_trt/
├── src/
│   ├── convert_data.py      # Convert VisDrone to YOLO format
│   ├── train.py              # Model training script
│   ├── export.py             # Export to TensorRT
│   └── infer_trt.py          # Real-time inference
├── data/
│   ├── visdrone_raw/         # Original VisDrone dataset
│   └── visdrone_yolo/        # Converted YOLO format
│       ├── images/           # train/val/test splits
│       ├── labels/           # YOLO format annotations
│       └── visdrone_config.yaml
├── models/
│   ├── engines/              # TensorRT engines
│   └── weights/              # PyTorch weights
|
├── Explore_Data.ipynb        # Dataset exploration
├── Benchmark_Report.ipynb    # Performance analysis
|
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.8+
- CUDA 11.8+ (for GPU acceleration)
- NVIDIA GPU with Compute Capability 7.0+ (RTX series recommended)
- conda or venv for environment management

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd drone_detection_trt
   ```

2. **Create and activate environment**
   ```bash
   conda create -n visdrone_env python=3.10
   conda activate visdrone_env
   ```

3. **Install dependencies**
   ```bash
   pip install ultralytics opencv-python pandas matplotlib seaborn jupyter
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

4. **Download VisDrone Dataset**
   - Download from [VisDrone Dataset](http://aiskyeye.com/)
   - Extract to `data/visdrone_raw/`

### Dataset Preparation

Convert VisDrone format to YOLO format:

```bash
python src/convert_data.py
```

This script:
- Converts bounding box format (VisDrone → YOLO normalized)
- Consolidates 10 classes into 3 meaningful categories
- Filters out ignored objects (truncation == 0)
- Splits data into train/val/test

**Class Mapping:**
- **Person**: pedestrian, people
- **Vehicle**: car, van, bus, truck
- **Cycle**: bicycle, tricycle, motorcycle, awning-tricycle

## Training

Train the YOLOv11s model:

```bash
python src/train.py
```

**Training Configuration:**
- Model: YOLOv11s (small variant)
- Epochs: 50
- Image Size: 640×640
- Batch Size: 8
- Device: GPU (CUDA)
- Optimizer: SGD with default YOLO settings

The best model will be saved to: `runs/detect/visdrone_final/weights/best.pt`

## TensorRT Export

Export the trained model to TensorRT for optimized inference:

```bash
python src/export.py
```

**Export Options:**
- Format: TensorRT engine
- Precision: FP16 (half precision for speed)
- Simplify: ONNX graph optimization enabled
- Device: GPU 0

Output: `runs/detect/visdrone_final/weights/best.engine`

> **Note**: First export takes 5-10 minutes as TensorRT optimizes for your specific GPU.

## Real-time Inference

Run real-time object detection on video:

```bash
python src/infer_trt.py
```

**Features:**
- Real-time FPS monitoring
- Inference latency display
- Bounding box visualization
- Press 'q' to quit

**Customize Video Source:**
```python
# In src/infer_trt.py
VIDEO_SOURCE = "test_video.mp4"  # or 0 for webcam
```

## Benchmarking

Explore dataset statistics and performance comparisons using Jupyter notebooks:

```bash
jupyter notebook
```

1. **Explore_Data.ipynb**: Dataset analysis
   - Image and annotation statistics
   - Class distribution
   - Bounding box size analysis
   - Sample visualizations

2. **Benchmark_Report.ipynb**: Performance comparison
   - PyTorch vs TensorRT metrics
   - Per-class performance
   - Processing pipeline breakdown
   - Speed/accuracy trade-offs

## Technical Details

### Data Conversion
The VisDrone dataset uses a specific annotation format:
```
<bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
```

Our converter:
1. Filters objects with `truncation == 0` (ignored regions)
2. Maps 10 original classes to 3 consolidated classes
3. Converts to YOLO format: `<class> <x_center> <y_center> <width> <height>` (normalized)

### Model Architecture
- **YOLOv11s**: Lightweight variant optimized for edge deployment
- **Input Size**: 640×640 pixels
- **Output**: 3 classes (Person, Vehicle, Cycle)
- **Anchor-free detection** with latest YOLO improvements

### TensorRT Optimization
- **FP16 Precision**: Uses Tensor Cores for 2x+ speedup
- **Kernel Fusion**: Combines operations to reduce memory transfers
- **Graph Optimization**: Removes unnecessary operations
- **Hardware-Specific**: Optimized for your GPU architecture

## Dataset Statistics

**VisDrone2019 Detection Dataset** (after conversion):
- **Training Set**: 6,471 images
- **Validation Set**: 548 images  
- **Test Set**: 1,610 images
- **Total Objects**: 540,000+ annotations
- **Image Resolution**: Variable (mostly 1920×1080 and 1360×765)

**Class Distribution:**
- Person: ~45% of annotations
- Vehicle: ~40% of annotations
- Cycle: ~15% of annotations

## Customization

### Training Different Models

```python
# In src/train.py
MODEL_NAME = 'yolo11n.pt'  # nano (fastest)
MODEL_NAME = 'yolo11s.pt'  # small (balanced) ✓ default
MODEL_NAME = 'yolo11m.pt'  # medium (more accurate)
```

### Adjusting Hyperparameters

```python
EPOCHS = 100           # More training
BATCH_SIZE = 16        # Larger batch (needs more VRAM)
IMG_SIZE = 1280        # Higher resolution (slower but more accurate)
```

### INT8 Quantization

For even faster inference (experimental):
```python
# In src/export.py
model.export(
    format='engine',
    device=0,
    int8=True,      # INT8 quantization
    data=DATA_CONFIG  # For calibration
)
```

## Troubleshooting

### CUDA Out of Memory
- Reduce `BATCH_SIZE` in `train.py`
- Use `yolo11n.pt` instead of `yolo11s.pt`
- Lower `IMG_SIZE` to 512 or 416

### TensorRT Export Fails
- Ensure CUDA and cuDNN are properly installed
- Check NVIDIA driver compatibility
- Try without `simplify=True`

### Low FPS During Inference
- Verify GPU is being used (check nvidia-smi)
- Ensure TensorRT engine matches your GPU architecture
- Close other GPU-intensive applications

## Citation

If you use the VisDrone dataset, please cite:

```bibtex
@article{zhu2021detection,
  title={Detection and tracking meet drones challenge},
  author={Zhu, Pengfei and Wen, Longyin and Du, Dawei and Bian, Xiao and Fan, Heng and Hu, Qinghua and Ling, Haibin},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume={44},
  number={11},
  pages={7380--7399},
  year={2021},
  publisher={IEEE}
}
}
```

## License

This project is provided as-is for educational and research purposes. Please respect the VisDrone dataset license terms.

## Acknowledgments

- **VisDrone Team** for the comprehensive drone dataset
- **Ultralytics** for the excellent YOLO implementation
- **NVIDIA** for TensorRT optimization framework

---

