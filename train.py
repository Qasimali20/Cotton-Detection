from ultralytics import YOLO

def main():
    
    model = YOLO("yolo12n.pt")

    results = model.train(
        data="data.yaml",    
        epochs=100,
        imgsz=640,
        batch=16,
        project="runs/detect",  
        device=0,               
    )

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()  # Optional, only needed if making a standalone .exe
    main()
