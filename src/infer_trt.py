import cv2
import time
from ultralytics import YOLO

MODEL_PATH = "runs/detect/visdrone_final/weights/best.engine"

VIDEO_SOURCE = "test_video_2.mp4"

def main():
    # Load the model
    print(f"Loading TensorRT Engine: {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)

    # Open video source
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"Error: Could not open video source {VIDEO_SOURCE}")
        return

    print("Starting Inference... Press 'q' to exit.")

    prev_time = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Run inference
        start_time = time.time()
        results = model(frame, verbose=False)
        end_time = time.time()

        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        annotated_frame = results[0].plot()

        inference_ms = (end_time - start_time) * 1000
        cv2.putText(annotated_frame, f"Mode: TensorRT FP16", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (20, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(annotated_frame, f"Latency: {inference_ms:.1f}ms", (20, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.imshow("Drone Surveillance System (RTX 4050)", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()