import json
import os

from PIL import Image
from torch.utils import data
from torchvision import transforms


class JsonDataSet(data.Dataset):
    def __init__(self, image_path, labels, transform) -> None:
        super().__init__()
        self.images = image_path
        self.labels = labels
        self.transforms = transform if transform else transforms.ToTensor()

    def __getitem__(self, index):
        img = self.images[index]
        label = self.labels[index]

        img = Image.open(img)
        img = self.transforms(img)

        return img, label

    def __len__(self):
        return len(self.images)


def default_train_transform(input_size):
    train_transforms = transforms.Compose([
        transforms.RandomRotation(degrees=20),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Resize(input_size),
        transforms.ToTensor(),
    ])
    return train_transforms


def default_test_transform(input_size):
    test_transforms = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
    ])
    return test_transforms


def get_dataset(data_path,
                train_json_file,
                test_json_file,
                size,
                train_transforms=None,
                test_transforms=None):
    train_json_file = os.path.join(data_path, train_json_file)
    test_json_file = os.path.join(data_path, test_json_file)
    train_json = json.load(open(train_json_file, 'r'))
    test_json = json.load(open(test_json_file, 'r'))

    train_img_lst = []
    train_label_lst = []
    for k, v in train_json['img'].items():
        train_img_lst.append(os.path.join(data_path, k))
        train_label_lst.append(int(v))

    test_img_lst = []
    test_label_lst = []
    for k, v in test_json['img'].items():
        test_img_lst.append(os.path.join(data_path, k))
        test_label_lst.append(int(v))

    if train_transforms:
        train_dataset = JsonDataSet(train_img_lst, train_label_lst,
                                    train_transforms)
    else:
        train_dataset = JsonDataSet(train_img_lst, train_label_lst,
                                    default_train_transform(size))
    if test_transforms:
        test_dataset = JsonDataSet(test_img_lst, test_label_lst,
                                   test_transforms)
    else:
        test_dataset = JsonDataSet(test_img_lst, test_label_lst,
                                   default_test_transform(size))

    return train_dataset, test_dataset
