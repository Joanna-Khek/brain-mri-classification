import os
import numpy as np
import shutil
import random
from pathlib import Path

from random import sample

import matplotlib.pyplot as plt
from PIL import Image

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from config import settings

def create_dataloaders(batch_size,
                       num_workers,
                       train_transform=None, 
                       test_transform=None):
    
    """Creates training and testing DataLoaders.
      
      Takes in a training directory and testing directory path and turns
      them into PyTorch Datasets and then into PyTorch DataLoaders.
      
      Args:
            train_dir: Path to training directory.
            test_dir: Path to testing directory.
            train_transform: torchvision transforms to perform on training set
            test_transform: torchvision transforms to perform on testing set
            batch_size: Number of samples per batch in each of the DataLoaders.
            num_workers: An integer for number of workers per DataLoader.
    """
    if train_transform is None:
        train_transform = transforms.ToTensor()

    if test_transform is None:
        test_transform = transforms.ToTensor()

    # Use ImageFolder to create datasets
    train_data = datasets.ImageFolder(root=settings.TRAIN_DIR,
                                      transform=train_transform,
                                      target_transform=None)

    valid_data = datasets.ImageFolder(root=settings.VALID_DIR,
                                      transform=train_transform,
                                      target_transform=None)

    test_data = datasets.ImageFolder(root=settings.TEST_DIR,
                                     transform=test_transform,
                                     target_transform=None)

    # Get class names
    class_names = train_data.classes

    # Turn images to data loaders
    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  shuffle=True)

    valid_dataloader = DataLoader(dataset=valid_data,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  shuffle=True)

    test_dataloader = DataLoader(dataset=test_data,
                                 batch_size=batch_size,
                                 num_workers=num_workers,
                                 shuffle=False)

    return train_dataloader, valid_dataloader, test_dataloader, class_names

def get_filenames(extensions: list, n: int, _class: str) -> Path:
    """Get full directory of the sampled images

    Args:
        extensions (list): Valid extensions (e.g jpeg, jpg)
        n (int): Number of sampled images
        _class (str): Class name

    Returns:
        Path: Paths of sample images
    """
    
    all_files = []
    for ext in extensions:
        path = settings.DATASET_DIR / settings.config['data_name'] / _class
        files = path.glob(ext)
        all_files.extend(files)
    
    sample_files = sample(all_files, n)
    return sample_files

def plot_sample_images(n: int, _class: str) -> None:
    sample_images = get_filenames(('*.jpeg', '*.jpg'), n, _class)

    rows = int(np.ceil(n/3))

    for num, x in enumerate(sample_images):
        img = Image.open(x)
        img_array = np.array(img)
        print('Array Dimensions', img_array.shape)
        plt.subplot(rows, 5, num+1)
        plt.title(x.name)
        plt.axis("off")
        plt.imshow(img)
    
    plt.tight_layout()
    plt.show()


def walk_through_dir(dir_path):
  """
  Walks through dir_path returning its contents.
  Args:
    dir_path (str or pathlib.Path): target directory
  
  Returns:
    A print out of:
      number of subdiretories in dir_path
      number of images (files) in each subdirectory
      name of each subdirectory
  """
  for dirpath, dirnames, filenames in os.walk(dir_path):
    print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

def move_images_to_directory(images: list,
                             label_path: Path,
                             final_path: Path) -> None:
    for img in images:
        image_path = label_path / img
        shutil.copy(image_path, final_path)

def split_dataset(train_set=0.8, valid_set=0.1):
    
    # Set up train, valid, test
    train_path = settings.TRAIN_DIR
    valid_path = settings.VALID_DIR
    test_path = settings.TEST_DIR

    train_path.mkdir(parents=True, exist_ok=True)
    valid_path.mkdir(parents=True, exist_ok=True)
    test_path.mkdir(parents=True, exist_ok=True)

    # Split Dataset
    dataset_path = Path(settings.DATASET_DIR, settings.config['data_name'])
    class_labels = os.listdir(dataset_path)

    for label in class_labels: # [no, yes]

        label_path = Path(dataset_path, label) # 'C:/..../brain_tunmor_dataset/no'
        list_of_images = os.listdir(label_path) # ['5 no.jpg', '6 no.jpeg', ..]
        total_num_images = len(list_of_images) # 250
        
        # shuffle list
        random.shuffle(list_of_images)

        # get the number of train images needed
        num_train = int(total_num_images * train_set)
        num_valid = int(total_num_images * valid_set)
        
        # split 
        train_images = list_of_images[:num_train]
        valid_images = list_of_images[num_train:num_train+num_valid]
        test_images = list_of_images[num_train+num_valid:]

        # final directory
        os.makedirs(train_path / label, exist_ok=True)
        os.makedirs(valid_path / label, exist_ok=True)
        os.makedirs(test_path / label, exist_ok=True)

        train_label_path = train_path / label
        valid_label_path = valid_path / label
        test_label_path = test_path / label

        # move the images
        move_images_to_directory(images=train_images,
                                 label_path=label_path,
                                 final_path=train_label_path)
        
        move_images_to_directory(images=valid_images,
                                 label_path=label_path,
                                 final_path=valid_label_path)
        
        move_images_to_directory(images=test_images,
                                 label_path=label_path,
                                 final_path=test_label_path)