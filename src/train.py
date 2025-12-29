from ultralytics import YOLO

MODEL_NAME = 'yolo11s.pt'
DATA_CONFIG = 'visdrone_config.yaml'
EPOCHS = 50  
IMG_SIZE = 640
BATCH_SIZE = 8
PROJECT_NAME = 'visdrone_final'

def train_model():
    # Load model
    print(f"Loading {MODEL_NAME}...")
    model = YOLO(MODEL_NAME)

    # Train
    print(f"Starting training for {EPOCHS} epochs...")
    results = model.train(
        data=DATA_CONFIG,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        device=0,           # Force GPU
        batch=BATCH_SIZE,
        project='runs/detect',
        name=PROJECT_NAME,
        exist_ok=True,
        plots=True
    )
    
    print(f"Training Complete. Best weights saved at: runs/detect/{PROJECT_NAME}/weights/best.pt")

if __name__ == '__main__':
    train_model()