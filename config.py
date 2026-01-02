import os  


class SystemConfig:
    HOST = "0.0.0.0"
    PORT = 5010
    DEBUG = False
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    LOG_DIR = os.path.join(BASE_DIR, 'logs')
    STATIC_DIR = os.path.join(BASE_DIR, 'static')
    UPLOAD_DIR = os.path.join('static', 'uploads')

    RECOGNITION_THRESHOLD = 0.7
    ANTI_FRAUD_COOLDOWN = 30

    IMAGE_SIZE = 112
    EMBEDDING_SIZE = 512
    BACKBONE = 'ResNet50'
    FACE_MODEL_PATH = 'ArcFace_model/weights.weights.h5'
    YOLO_MODEL_PATH = 'yolov8n.pt'
