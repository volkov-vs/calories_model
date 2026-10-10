# Конфигурация

class Config:
    # настройки загрузки данных
    ROOT_IMAGES_DIR = "data/images/"
    DISHES_FILE = "data/dish.csv"

    # настройки параметров преобразований
    RANDOM_HORIZONTAL_FLIP = 0.2
    RANDOM_VERTICAL_FLIP = 0.2
    RANDOM_ROTATION = 5

    # настройки процесса обучения
    BATCH_SIZE = 32
    EPOCHS = 100

    IMAGE_MODEL_LR = 0.0001
    INGREDIENTS_MODEL_LR = 0.0001
    FUSION_HEAD_LR = 0.0001

    # настройки сохранения модели
    SAVE_MODEL_PATH = "models/best_multimodal_model.pth"