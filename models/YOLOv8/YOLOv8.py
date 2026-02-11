from ultralytics import YOLO
import os


MODEL_NAME = 'yolov8m-cls.pt'  # yolov8n-cls.pt, yolov8s-cls.pt, etc.
PRETRAINED = True 

DATASET_PATH = os.path.abspath('/root/project/dataset/dataset_cropped') 

EPOCHS = 100
BATCH_SIZE = 256
IMG_SIZE = 224
LEARNING_RATE = 3e-4
OPTIMIZER = 'AdamW'
WARMUP_EPOCHS = 3.0
LRF = 0.01
COS_LR = True
PATIENCE = 20

WEIGHT_DECAY = 0.01
DROPOUT = 0.0
LABEL_SMOOTING = 0.0
AUTO_AUGMENT = None # 'randaugment'
ERASING = 0.0
MOSAIC = 0.0
MIXUP=0.0
CUTMIX=0.0
LS=0.0

DEVICE = 'cuda:0'        
WORKERS = 16

PROJECT_NAME = 'YOLOv8_Cls'
RUN_NAME = 'Yolos1'
SAVE_PERIOD = 10
CACHE = False
SEED = 0

def main():
    print(f"[Info] Loading Model: {MODEL_NAME}")
    model = YOLO(MODEL_NAME) 

    print(f"[Info] Starting Training on {DEVICE}...")
    results = model.train(
        data=DATASET_PATH,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        lr0=LEARNING_RATE,
        optimizer=OPTIMIZER,
        warmup_epochs=WARMUP_EPOCHS,
        lrf=LRF,
        cos_lr=COS_LR,
        patience=PATIENCE,
        weight_decay=WEIGHT_DECAY,
        dropout=DROPOUT,
        label_smoothing=LABEL_SMOOTING,
        auto_augment=AUTO_AUGMENT,
        erasing=ERASING,
        workers=WORKERS,
        save_period=SAVE_PERIOD,
        cache=CACHE,
        mosaic=MOSAIC,
        seed=SEED,
        device=DEVICE,
        project=PROJECT_NAME,
        name=RUN_NAME,
        pretrained=PRETRAINED,
        exist_ok=True,
        fliplr=0.5,
        flipud=0.5,          
        degrees=45.0,        
        hsv_h=0.01,          
        hsv_s=0.10,
        hsv_v=0.05,
        mixup=MIXUP,
        cutmix=CUTMIX,
        label_smoothing=LS,
        plots=True, 
        val=True,                
        verbose=True,            
    )

    print("[Info] Training Completed.")
    print(f"Results saved to runs/{PROJECT_NAME}/{RUN_NAME}")

if __name__ == '__main__':
    main()
