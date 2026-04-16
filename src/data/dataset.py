import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from src.utils.util import read_data, one_hot
from src.utils.config import CONFIG


class IMGDataset(Dataset):

    def __init__(self, data_path: str, split: str, resol: int = 256):
        assert split in ['train', 'test']
        self.data = read_data(data_path, split)

        if 'cub' in data_path:
            self.n_class = CONFIG['cub']['N_CLASSES']
        elif 'awa' in data_path:
            self.n_class = CONFIG['awa']['N_CLASSES']

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.transform = img_augment(split=split, resol=resol, mean=mean, std=std)
        self._set()

    def _set(self, data=None):
        self.image_path = []
        self.concept = []
        self.label = []
        self.one_hot_label = []

        if data is None:
            data = self.data

        for instance in data:
            label = instance['label']
            self.image_path.append(instance['img_path'])
            self.concept.append(instance['concept'])
            self.label.append(label)
            self.one_hot_label.append(one_hot(label, self.n_class))

    def reset(self):
        self.image_path = []
        self.concept = []
        self.label = []
        self.one_hot_label = []

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, index):
        image = Image.open(self.image_path[index])
        image = self.transform(image.convert("RGB"))
        concept = torch.tensor(self.concept[index], dtype=torch.float32)
        label = torch.tensor(self.label[index])
        one_hot_label = torch.tensor(self.one_hot_label[index])
        return image, concept, label, one_hot_label


def get_dataloader(data_path, batch_size):
    train_loader = DataLoader(
        dataset=IMGDataset(data_path, split='train'),
        batch_size=batch_size,
        shuffle=True,
        num_workers=16
    )
    test_loader = DataLoader(
        dataset=IMGDataset(data_path, split='test'),
        batch_size=batch_size,
        shuffle=False,
        num_workers=16
    )
    return train_loader, test_loader


def img_augment(split, resol, mean, std):
    if split == 'train':
        return transforms.Compose([
            transforms.ColorJitter(brightness=32/255, saturation=(0.5, 1.5)),
            transforms.RandomResizedCrop(resol),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:
        return transforms.Compose([
            transforms.CenterCrop(resol),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
