import os
import sys
sys.path.append('.')
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

from data.transform import Onehot, encode_onehot

# sample classes not in cifar-10
label_list = [0,4,5,6,7,8,13,14,15,28,30,31,32,33,34,35,36,37,38,40,41,42,43,44,45,46,48,49,50,51,52,53,54,55,
            56,57,58,59,60,61,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,
            90,91,92,94,95,96,97,98,99]

imagenet100_labels = {0: 'cock', 1: 'goldfinch', 2: 'indigo_bunting', 3: 'great_grey_owl', 4: 'European_fire_salamander', 5: 'spotted_salamander', 6: 'loggerhead', 7: 'agama', 8: 'Gila_monster',9: 'ptarmigan', 
                    10: 'prairie_chicken', 11: 'peacock', 12: 'bee_eater', 13: 'jellyfish', 14: 'American_lobster', 15: 'crayfish', 16: 'crane', 17: 'bustard', 18: 'albatross', 19: 'Shih-Tzu', 
                    20: 'bloodhound', 21: 'whippet', 22: 'flat-coated_retriever', 23: 'Sussex_spaniel', 24: 'Border_collie', 25: 'Doberman', 26: 'Greater_Swiss_Mountain_dog', 27: 'Great_Pyrenees', 28: 'dingo', 29: 'cougar', 
                    30: 'snow_leopard', 31: 'leaf_beetle', 32: 'rhinoceros_beetle', 33: 'cicada', 34: 'lacewing', 35: 'damselfly', 36: 'cabbage_butterfly', 37: 'Angora', 38: 'hamster', 39: 'sorrel', 
                    40: 'hippopotamus', 41: 'weasel', 42: 'gorilla', 43: 'langur', 44: 'colobus', 45: 'howler_monkey', 46: 'airliner', 47: 'ashcan', 48: 'Band_Aid', 49: 'baseball', 
                    50: 'beacon', 51: 'bow_tie', 52: 'brassiere', 53: 'candle', 54: 'car_wheel', 55: 'chain_mail', 56: 'crash_helmet', 57: 'crib', 58: 'drumstick', 59: 'fire_screen', 
                    60: 'football_helmet', 61: 'four-poster', 62: 'garbage_truck', 63: 'golf_cart', 64: 'gondola', 65: 'greenhouse', 66: 'half_track', 67: 'harmonica', 68: 'honeycomb', 69: 'hook', 
                    70: 'knee_pad', 71: 'ladle', 72: 'loudspeaker', 73: 'maillot', 74: 'marimba', 75: 'maypole', 76: 'miniskirt', 77: 'mortarboard', 78: 'obelisk', 79: 'piggy_bank', 
                    80: 'pool_table', 81: 'pot', 82: 'reel', 83: 'revolver', 84: 'rotisserie', 85: 'salt_shaker', 86: 'shower_curtain', 87: 'spindle', 88: 'stove', 89: 'swimming_trunks', 
                    90: 'swing', 91: 'tank', 92: 'toilet_seat', 93: 'tow_truck', 94: 'tub', 95: 'wig', 96: 'acorn_squash', 97: 'Granny_Smith', 98: 'cliff', 99: 'promontory'}

def load_data(root, batch_size, workers):
    """
    Load imagenet dataset

    Args:
        root (str): Path of imagenet dataset.
        batch_size (int): Number of samples in one batch.
        workers (int): Number of data loading threads.

    Returns:
        train_loader (torch.utils.data.DataLoader): Training dataset loader.
        query_loader (torch.utils.data.DataLoader): Query dataset loader.
        val_loader (torch.utils.data.DataLoader): Validation dataset loader.
    """
    # Data transform
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    query_retrieval_init_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

    # Construct data loader
    train_dir = os.path.join(root, 'train')
    query_dir = os.path.join(root, 'query')
    retrieval_dir = os.path.join(root, 'database')

    train_dataset = ImagenetDataset(
        train_dir,
        transform=train_transform,
        target_transform=Onehot()
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
    )

    query_dataset = ImagenetDataset(
        query_dir,
        transform=query_retrieval_init_transform,
        target_transform=Onehot(),
    )

    query_loader = DataLoader(
        query_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )

    retrieval_dataset = ImagenetDataset(
        retrieval_dir,
        transform=query_retrieval_init_transform,
        target_transform=Onehot(),
    )

    retrieval_loader = DataLoader(
        retrieval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )

    return train_loader, query_loader, retrieval_loader


class ImagenetDataset(Dataset):
    classes = None
    class_to_idx = None

    def __init__(self, root, transform=None, target_transform=None, label_list=label_list):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.data = []
        self.targets = []

        # Assume file alphabet order is the class order
        if ImagenetDataset.class_to_idx is None:
            ImagenetDataset.classes, ImagenetDataset.class_to_idx = self._find_classes(root)

        for i, cl in enumerate(ImagenetDataset.classes):
            if ImagenetDataset.class_to_idx[cl] in label_list:
                cur_class = os.path.join(self.root, cl)
                files = os.listdir(cur_class)
                files = [os.path.join(cur_class, i) for i in files]
                self.data.extend(files)
                self.targets.extend([ImagenetDataset.class_to_idx[cl] for i in range(len(files))])
        self.data = np.array(self.data)
        self.targets = np.array(self.targets)
        self.onehot_targets = encode_onehot(self.targets, 100)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        img, target = self.data[item], self.targets[item]

        img = Image.open(img).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        
        if self.target_transform is not None:
            target = self.target_transform(target, 100)
        return img, target, item

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx
    
    def get_onehot_targets(self):
        '''
        Return one-hot encoding targets.
        '''
        return torch.from_numpy(self.onehot_targets).float()

