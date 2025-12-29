from ultralytics import YOLO
import torch

def main():
    # Load the YOLOv8 Nano model
    model = YOLO('yolov8n.pt')

    # Start training
    # We use device=0 for GPU, or 'cpu' for a quick login-node test
    results = model.train(
        data='sds_data.yaml',
        epochs=100,
        imgsz=640,
        device=0  # Change to 'cpu' if testing right now on login node
    )

if __name__ == '__main__':
    main()