import torch
from torch import nn
import torchvision

import pandas as pd
import numpy as np

from tqdm import tqdm
from datetime import datetime

from dataset import build_dictionary, get_data_from_dataframe, get_mean_std, DishesDataset, plot_images


from torchvision.transforms.v2 import RandomRotation, RandomHorizontalFlip, RandomVerticalFlip, Compose, Resize, ToTensor, Normalize
from torch.utils.data import DataLoader
from torchmetrics import MeanAbsoluteError
from torchvision.models import efficientnet_b3

from sklearn.model_selection import train_test_split

from torch import optim

import gc

import matplotlib.pyplot as plt


#--------------------------------------------------------------------------------------------------
class CaloriesModel(nn.Module):
    def __init__(self, ingredients_number):
        super().__init__()

        # модель для изображений        
        self.image_model = efficientnet_b3(weights='IMAGENET1K_V1')
        self.image_model.classifier = nn.Sequential(
            nn.Linear(1536, 1536),
            nn.ReLU()
        )

        # заморозка слоёв модели
        for param in self.image_model.parameters():
            param.requires_grad = False

        # разморозка полносвязного слоя classifier
        for param in self.image_model.classifier.parameters():
            param.requires_grad = True 
        
        self.ingredients_model = nn.Sequential(
            nn.Linear(ingredients_number, 256),
            nn.ReLU()
        )
        
        self.fusion_head = nn.Sequential(
            nn.Linear(1536 + 256, 512),  # 1536 = EfficientNet-B3 out
            nn.ReLU(),
            nn.Linear(512, 256),            
            nn.ReLU(),            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
    
    def forward(self, image, ingredients):
        img_emb = self.image_model(image)
        calories_emb = self.ingredients_model(ingredients)
        fused = torch.cat([img_emb, calories_emb], dim=1)
        return self.fusion_head(fused).squeeze(1)

#--------------------------------------------------------------------------------------------------
# обучение одной эпохи
def train_one_epoch(epoch, model, train_loader, device, optimizer, criterion, mae_metric):
    
    model.train(True)

    running_loss = 0.0
    # mean_absolute_error
    mae = 0.0
    
    for item in tqdm(train_loader, desc=f"Train     "):
        # Извлечение батча
        images, ingredients, total_calories = item[0], item[1], item[2]
        
        # перенос данных на устройство обучения
        images = images.to(device)
        ingredients = ingredients.to(device)
        total_calories = total_calories.to(device)

        # Обнуление градиентов
        optimizer.zero_grad()

        # Прямое распространение        
        predicted_calories = model(images, ingredients)
        
        # Подсчёт ошибки
        loss = criterion(predicted_calories, total_calories)

        # Обратное распространение
        loss.backward()

        # Обновление весов
        optimizer.step()

        # Суммирование ошибки
        running_loss += loss.item()

        #! накопление абсолютной ошибки
        mae += mae_metric(predicted_calories, total_calories)

    # освобождение памяти
    if device == "cuda":
        torch.cuda.empty_cache()

    gc.collect()

    return running_loss / len(train_loader), mae / len(train_loader)

#--------------------------------------------------------------------------------------------------
# Валидация модели
def validate(epoch, model, val_loader, device, criterion, mae_metric):
    # перевод модели в режим валидации
    model.eval()
    
    running_loss = 0.0
    # mean_absolute_error
    mae = 0.0
    
    with torch.no_grad():
            
        for item in tqdm(val_loader, desc=f"Validation"):
            # image, ingredients, total_calories
            images, ingredients, total_calories = item[0], item[1], item[2]        
            
            # перенос данных на устройство обучения
            images = images.to(device)
            ingredients = ingredients.to(device)
            total_calories = total_calories.to(device)

            # расчет выхода модели    
            predicted_calories = model(images, ingredients)

            loss = criterion(predicted_calories, total_calories)
            
            running_loss += loss

            #! накопление абсолютной ошибки
            mae += mae_metric(predicted_calories, total_calories)        
    
    # освобождение памяти
    if device == "cuda":
        torch.cuda.empty_cache()

    gc.collect()

    return running_loss / len(val_loader), mae / len(val_loader)

def calculate_metrics(desc, model, data_loader, device, criterion, mae_metric):
    # перевод модели в режим валидации
    model.eval()
    
    running_loss = 0.0
    # mean_absolute_error
    mae = 0.0
    
    with torch.no_grad():
            
        for item in tqdm(data_loader, desc=desc):
            # image, ingredients, total_calories
            images, ingredients, total_calories = item[0], item[1], item[2]        
            
            # перенос данных на устройство обучения
            images = images.to(device)
            ingredients = ingredients.to(device)
            total_calories = total_calories.to(device)

            # расчет выхода модели    
            predicted_calories = model(images, ingredients)

            loss = criterion(predicted_calories, total_calories)
            
            running_loss += loss

            #! накопление абсолютной ошибки
            mae += mae_metric(predicted_calories, total_calories)        
    
    # освобождение памяти
    if device == "cuda":
        torch.cuda.empty_cache()

    gc.collect()

    return running_loss / len(data_loader), mae / len(data_loader)

#--------------------------------------------------------------------------------------------------
# Отображает историю
def plot_history(history):    
    epochs_values = range(1, history["epochs"] + 1)

    _ , axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

    # График ошибки
    axes[0].plot(epochs_values, history["train_loss"], label="Train Loss", color="blue", linewidth=2)
    axes[0].plot(epochs_values, history["val_loss"], label="Val Loss", color="red", linewidth=2)
    axes[0].set_title("Training and Validation Loss")
    axes[0].legend()

    # График точности
    axes[1].plot(epochs_values, history["train_mae"], label="Train MAE", color="blue", linewidth=2)
    axes[1].plot(epochs_values, history["val_mae"], label="Val   MAE", color="red", linewidth=2)
    axes[1].set_title("Training and Validation MAE")
    axes[1].legend()
    
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.show()



#--------------------------------------------------------------------------------------------------
# обучение модели
def train(config):
    #! Этап 1. Загрузка и подготовка данных
    
    dishes_df = pd.read_csv(config.DISHES_FILE)
    ingredients_dict = build_dictionary(dishes_df, "ingredients")

    # загрузка обучающих данных
    masses_train, total_calories_list, structures_list, images_filenames_list = get_data_from_dataframe(dishes_df, ingredients_dict, split_value="train")

    # разделение выобучающих данных на обучающие и валидационные
    total_calories_train, total_calories_val, structures_train, structures_val, images_filenames_train, images_filenames_val = train_test_split(
        total_calories_list, structures_list, images_filenames_list,
        test_size=0.2,   # 20% на валидацию
        random_state=42, 
        shuffle=True     
    )

    # загрузка тестовых данных
    masses_test, total_calories_test, structures_test, images_filenames_test = get_data_from_dataframe(dishes_df, ingredients_dict, split_value="test")

    print("Количество обучающих данных    : ", len(images_filenames_train))
    print("Количество валидационных данных: ", len(images_filenames_val))
    print("Количество тестовых данных     : ", len(images_filenames_test))

    #max_target = total_calories_train.max()

    print("Оценка средних и стандартных отклонений по каналам ...")
    # оценка средних и стандартных отклонений по каналам
    mean, std = get_mean_std(config.ROOT_IMAGES_DIR, images_filenames_train)

    print("Оценка средних и стандартных отклонений по каналам завершена.")
    print("mean:")
    print(mean)

    print("std:")
    print(std)

    #--------------------------------------------------------------------------------------------------
    #! Этап 2. Создание датасета

    # преобразования обучающего датасета
    train_transforms = Compose([        
        Resize((320, 320)),   # для модели b3        
        RandomHorizontalFlip(p=config.RANDOM_HORIZONTAL_FLIP),
        RandomVerticalFlip(p=config.RANDOM_VERTICAL_FLIP),
        RandomRotation([-config.RANDOM_ROTATION, config.RANDOM_ROTATION]),
        ToTensor(),        
        Normalize(mean, std)
    ])

    # преобразования тестового датасета
    val_transforms = Compose([        
        Resize((320, 320)),   # для модели b3        
        ToTensor(),        
        Normalize(mean, std)
    ])

    # преобразование датасетов
    print("Создание обучающего датасета")
    begin = datetime.now()
    train_dataset = DishesDataset(total_calories_train, structures_train, 
                              config.ROOT_IMAGES_DIR, images_filenames_train, train_transforms)
    duration = datetime.now() - begin
    print("Продолжительность создания датасета: ", duration)

    print("Создание валидационного датасета")
    begin = datetime.now()
    val_dataset = DishesDataset(total_calories_val, structures_val, 
                              config.ROOT_IMAGES_DIR, images_filenames_val, val_transforms)
    duration = datetime.now() - begin
    print("Продолжительность создания датасета: ", duration)
    print()

    print("Создание тестового датасета")
    begin = datetime.now()
    test_dataset = DishesDataset(total_calories_test, structures_test, 
                                  config.ROOT_IMAGES_DIR, images_filenames_test, val_transforms)
    duration = datetime.now() - begin
    print("Продолжительность создания датасета: ", duration)
    print()

    # количество классов для обучения = размер словаря ингредиентов
    ingredients_number = len(ingredients_dict)

    print("Создание объектов DataLoader")

    batch_size = config.BATCH_SIZE

    begin = datetime.now()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    duration = datetime.now() - begin
    print("Продолжительность создания объектов DataLoader: ", duration)
    print()
    print("train dataloader: ", len(train_loader))
    print("val   dataloader: ", len(val_loader))
    print("test  dataloader: ", len(test_loader))

    #--------------------------------------------------------------------------------------------------
    #! Этап 3. Объявление модели

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #
    calories_model = CaloriesModel(ingredients_number).to(device=device)
    
    print("Количество параметров в модели калорийности:", sum(p.numel() for p in calories_model.parameters()))

    #--------------------------------------------------------------------------------------------------
    #! Этап 4. Обучение модели и валидация
    #     
    criterion = nn.MSELoss().to(device)

    # Оптимизатор с разными LR и регуляризацией
    optimizer = optim.AdamW([
        {'params': calories_model.image_model.parameters(), 'lr': config.IMAGE_MODEL_LR, 'weight_decay': 0.001},
        {'params': calories_model.ingredients_model.parameters(), 'lr': config.INGREDIENTS_MODEL_LR, 'weight_decay': 0.001},
        {'params': calories_model.fusion_head.parameters(), 'lr': config.FUSION_HEAD_LR, 'weight_decay': 0.001}
    ])

    # расчет ошибки Mean Absolute Error
    mae_metric = MeanAbsoluteError().to(device)    

    # информация об обучении
    train_history = {
        "epochs" : config.EPOCHS,
        "train_loss" : [],
        "val_loss" : [],       
        "train_mae" : [],
        "val_mae" : []
    }

    best_val_loss = 1e5

    begin = datetime.now()   

    for epoch in range(config.EPOCHS):
        print(f"> Эпоха {epoch + 1}")
        # обучение для одной эпохи 
        train_avg_loss, train_mae = train_one_epoch(epoch, calories_model, train_loader, device, optimizer, criterion, mae_metric)

        # валидация модели        
        val_avg_loss, val_mae = calculate_metrics("Validation", calories_model, val_loader, device, criterion, mae_metric)
    
        # код сохранения лучшей модели
        if val_avg_loss < best_val_loss:
            best_val_loss = val_avg_loss            
        
            torch.save(calories_model.state_dict(), config.SAVE_MODEL_PATH)
        
        print(f"Train Loss: {train_avg_loss}, Val Loss: {val_avg_loss}, Best Val Loss {best_val_loss}")
        print(f"Train MAE : {train_mae}, Val MAE : {val_mae}")

        # добавление данных в историю
        train_history["train_loss"].append(train_avg_loss)
        train_history["train_mae"].append(train_mae.item())

        train_history["val_loss"].append(val_avg_loss.item())
        train_history["val_mae"].append(val_mae.item())


    duration = datetime.now() - begin
    print("Продолжительность обучения: ", duration)

    #! Этап 5. Расчет метрик на тестовых данных
    test_avg_loss, test_mae = calculate_metrics("Test      ", calories_model, test_loader, device, criterion, mae_metric)
    print(f"Test Loss {test_avg_loss}")
    print(f"Test MAE  {test_mae}")

    plot_history(train_history)

#--------------------------------------------------------------------------------------------------
# Использование модели
def use_model(config):
    # загрузка данных
    dishes_df = pd.read_csv(config.DISHES_FILE)
    ingredients_dict = build_dictionary(dishes_df, "ingredients")

    classes_number = len(ingredients_dict)

    # загрузка тестовых данных
    masses_test, total_calories_test, structures_test, images_filenames_test = get_data_from_dataframe(dishes_df, ingredients_dict, split_value="test")

    # преобразования тестового датасета
    val_transforms = Compose([
        Resize((320, 320)),   # для модели b3
        ToTensor(),
        Normalize(
            (0.48810631036758423, 0.4708237946033478, 0.43778884410858154), # параметры из предыдущего этапа
            (0.22308269143104553, 0.2369823306798935, 0.26331982016563416))
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dataset = DishesDataset(total_calories_test, structures_test, 
                              config.ROOT_IMAGES_DIR, images_filenames_test, val_transforms)

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Загрузка модели
    calories_model = CaloriesModel(classes_number)
    model_file = config.SAVE_MODEL_PATH

    calories_model.load_state_dict(torch.load(model_file, weights_only=True))
    calories_model.to(device)
    calories_model.eval()

    # инференс
    max_count = 5
    index = 0

    dish_list = []

    print()
    print("Модель: ", model_file)
    print("Примеры расчета калорийности блюд: ")

    sum_error = 0

    mae_metric = MeanAbsoluteError().to(device)
            
    for item in test_loader:
        # image, ingredients, total_calories
        images, ingredients, total_calories = item[0], item[1], item[2]        
            
        # перенос данных на устройство обучения
        images = images.to(device)
        ingredients = ingredients.to(device)
        total_calories = total_calories.to(device)

        # расчет выхода модели    
        predicted_calories = calories_model(images, ingredients)

        #расчет абсолютной ошибки
        absolute_error = mae_metric(predicted_calories, total_calories)

        sum_error += absolute_error

        dish_list.append((images_filenames_test[index], total_calories.item(), predicted_calories.item(), absolute_error.item()))

        # выводим только первые 5
        if index < max_count:
            print(f"{images_filenames_test[index]}: true calories = {total_calories.item():.3f} predicted calories = {predicted_calories.item():.3f}")
    
        index = index + 1

    # сортировка списка
    sorted_dishes = sorted(dish_list, key=lambda dish: dish[3])

    # вывод 5 блюд, для которых ошибка наибольшая
    print()
    top_count = 5
    print(f"{top_count} блюд с наибольшей ошибкой:")    
    i = 0
    count = len(sorted_dishes)

     # список изображений
    images_filenames = []
    for index in range(top_count):
        dish = sorted_dishes[count - index - 1]
        print(f"{index + 1}. {dish[0]}: true calories = {dish[1]:.3f} predicted calories = {dish[2]:.3f} absolute_error = {dish[3]:.3f}")
        images_filenames.append(dish[0])   
    
    images_filenames.append(images_filenames[-1]) # фиктивный, чтобы не переделывать функцию

    plot_images(config.ROOT_IMAGES_DIR, images_filenames, num_rows=2, num_cols=3)    