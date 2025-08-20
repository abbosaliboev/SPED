from ultralytics import YOLO

if __name__ == "__main__":

    model = YOLO("yolov8n.yaml")

 
    model.train(
        data="C:/Users/dalab/Desktop/Abbos/SmartLight/dataset/data.yaml",
        epochs=100,              
        imgsz=640,              
        batch=32,               
        device=0,               
        name="assistive-detect",
        project="runs2",  
        pretrained=True,         
        workers=8,               
        verbose=False            
    )
