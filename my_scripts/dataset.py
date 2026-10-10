import torch
from torchvision.transforms.functional import to_tensor
from torch.utils.data import Dataset

from PIL import Image

import pandas as pd

import matplotlib.pyplot as plt
import numpy as np

import os

#------------------------------------------------------------------------------------
# Получает датафрейм по значению split
def get_sample_df(data_frame, split_value="train"):
    
    mask = data_frame.loc[:, "split"] == split_value

    filtered = data_frame[mask]

    return filtered

#------------------------------------------------------------------------------------
# Выводит информацию о датафрейме
def describe_dataframe(dataframe, first_rows=10, last_rows=10):
    dataframe.describe()

    print("Описательная статистика:")
    print(dataframe.describe())

    print()
    print(f"Первые {first_rows} строк")
    print(dataframe.head(first_rows))
    print()
    print(f"Последние {last_rows} строк")
    print(dataframe.tail(last_rows))

# выполняет построение словаря ингредиентов
def build_dictionary(dataframe, column_name, separator=';'):
    dictionary = {}

    index = 0

    for _, row in dataframe.iterrows():
        items = row[column_name].split(separator)

        for item in items:
            key = item.strip()
            if key not in dictionary.keys():
                dictionary[key] = index
                index = index + 1

    return dictionary

#------------------------------------------------------------------------------------
# получает данные из датафрейма pandas
def get_data_from_dataframe(dataframe, ingredients_dict, split_value, separator=';'):

    # список массы блюда
    masses = []
    # список калорийностей блюд
    total_calories = []
    # список составов блюд
    structure_list = []
    # список имен файлов изображений блюд
    images_filenames = []

    for index, row in dataframe.iterrows():    
        if row["split"] != split_value:
            continue

        # масса блюда
        masses.append(float(row["total_mass"]))

        # калорийность блюда
        total_calories.append(float(row["total_calories"]))

        #  состав блюда по наличию ингредиентов
        ingredients = row["ingredients"].split(separator)
        
        structure = []
        for ingredient in ingredients:
            structure.append(ingredient.strip())

        structure_list.append(structure)

        # имена файлов изображений блюд
        dish_id = row["dish_id"]
        images_filenames.append(f"{dish_id}/rgb.png")

    # преобразовать списки в тензоры
    masses_tensor = torch.tensor(masses)
    total_calories_tensor = torch.tensor(total_calories)

    structure_size = len(ingredients_dict)
    structures_tensor = torch.zeros(len(structure_list), structure_size)
    
    for i in range(len(structure_list)):
        for ingredient_id in structure_list[i]:
            ingredient_index = ingredients_dict[ingredient_id]
            
            structures_tensor[i, ingredient_index] = 1.0

    return masses_tensor, total_calories_tensor, structures_tensor, images_filenames

#------------------------------------------------------------------------------------
# определяет среднее и стандартное отклонение
def get_mean_std(root_dir, image_filenames):

    images = []

    data = torch.zeros(len(image_filenames), 3, 480, 640)

    for index, image_filename in enumerate(image_filenames):
        image_path = os.path.join(root_dir, image_filename)
        image = Image.open(image_path)
        images.append(image)

        image_tensor = to_tensor(image)

        data[index] = image_tensor    

    mean = [0.0, 0.0, 0.0]
    std = [0.0, 0.0, 0.0]

    for c in range(3):
        mean[c] = torch.mean(data[:, c]).item()
        std[c] = torch.std(data[:, c]).item()

    return (mean[0], mean[1], mean[2]), (std[0], std[1], std[2])

#------------------------------------------------------------------------------------
# Показывает изображения
def plot_images(images_dir, image_filenames, num_rows=2, num_cols=2):
    
    image_paths = [os.path.join(images_dir, filename) for filename in image_filenames[:(num_rows *num_cols)]]

    images = [Image.open(path) for path in image_paths]

    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(10,10))

    index = 0
    
    for i in range(num_rows):
        for j in range(num_cols):
            # Преобразование в массив
            img_array = np.array(images[index])

            axes[i, j].imshow(img_array)
            axes[i, j].set_title(image_filenames[index])
            axes[i, j].axis("off")

            index += 1

    plt.tight_layout()
    plt.show()

#------------------------------------------------------------------------------------
# Показать информацию о данных
def show_data_info(images_dir, dishes_file):
    dishes_df = pd.read_csv(dishes_file)
    ingredients_dict = build_dictionary(dishes_df, "ingredients")

    print("Размер словаря игредиентов:", len(ingredients_dict))
    describe_dataframe(dishes_df)

    # загрузка обучающих данных
    masses_train, total_calories_train, structures_train, images_filenames_train = get_data_from_dataframe(dishes_df, ingredients_dict, split_value="train")
    max_target = total_calories_train.max()

    print("Обучающие данные загружены.")    

    print("Примеры изображений:")
    plot_images(images_dir, images_filenames_train, num_rows=4, num_cols=3)

    # Построение гистограммы распределения массы блюд
    plt.figure(figsize=(10, 4))
    plt.hist(masses_train, bins=30, color='lightblue')
    
    plt.title("Гистограмма распределения массы блюд")
    plt.xlabel("Масса блюд")
    plt.ylabel("Количество блюд")
    plt.grid(axis='y',linestyle='--', alpha=0.3)
    
    plt.show()

    # Построение гистограммы распределения калорийности блюд
    plt.figure(figsize=(10, 4))
    plt.hist(total_calories_train, bins=30, color='lightblue')
        
    plt.title("Гистограмма распределения калорийности блюд")
    plt.xlabel("Калорийность блюд")
    plt.ylabel("Количество блюд")
    plt.grid(axis='y',linestyle='--', alpha=0.3)
        
    plt.show()

    print()

# Класс для датасета
class DishesDataset(Dataset):
  def __init__(self, total_calories, dish_structures, root_dir, image_filenames, transforms):
    super(Dataset, self).__init__()
    self.total_calories = [torch.tensor(item) for item in total_calories]
    self.dish_structures = [torch.tensor(item) for item in dish_structures]
    self.transforms = transforms

    # загрузить изображения в память    
    self.images = []
    for image_filename in image_filenames:
        image_path = os.path.join(root_dir, image_filename)
        image = Image.open(image_path)
        self.images.append(image)

  def __len__(self):
    return len(self.total_calories)

  def __getitem__(self, idx):
    total_calories = self.total_calories[idx]

    transformed_image = self.transforms(self.images[idx])
    dish_structure = self.dish_structures[idx]
    
    return transformed_image, dish_structure, total_calories

    