import skimage.io as io
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class PolypDataset(Dataset):
    def __init__(self, data_file, label_file, input_size):
        with open(data_file) as f:
            self.image_paths = f.readlines()
        f.close()

        with open(label_file) as f:
            self.label_paths = f.readlines()
        f.close()

        self.image_paths = [x.strip() for x in self.image_paths]
        self.label_paths = [x.strip() for x in self.label_paths]

        self.input_size = input_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        label = io.imread(self.label_paths[idx])
        label = self.preprocess_label()(label)

        img = io.imread(self.image_paths[idx])
        img = self.preprocess()(img)
        return (img, label, idx)

    def preprocess(self):
        return transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(self.input_size, Image.BICUBIC),
                transforms.ToTensor(),
            ]
        )

    def preprocess_label(self):
        return transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(self.input_size, Image.BICUBIC),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
            ]
        )


input_size = (128, 128)

# Create train and validation sets
train_images_file = "/home/raman/Downloads/doubleunet/data/train_images.txt"
train_labels_file = "data/train_masks.txt"
val_images_file = "data/val_images.txt"
val_labels_file = "data/val_masks.txt"

# Initialise Datasets
train_set = PolypDataset(train_images_file, train_labels_file, input_size)
val_set = PolypDataset(val_images_file, val_labels_file, input_size)

# Batch size for training
batch_size = 8

# Random seed for dataloader
random_seed = 7

# Initialize Dataloaders
torch.manual_seed(random_seed)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
print("Data Loaded.")
