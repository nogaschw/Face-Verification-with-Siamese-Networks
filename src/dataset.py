import torch
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score
from torch.nn.functional import pairwise_distance

class SiameseDataset(Dataset):
    def __init__(self, df):
        """
        pairs: list of (img1_path, img2_path, label)
        transform: optional image transform (e.g., resizing, normalization)
        """
        self.df = df
        self.clean_transform = transforms.Compose([
            transforms.Resize((105, 105)),
            transforms.ToTensor()
        ])
        self.noise_transform = transforms.Compose([
            transforms.Resize((105, 105)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.clamp(x + 0.3 * torch.randn_like(x), 0., 1.))
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_1 = self.clean_transform(Image.open(get_the_image(row['name_1'], row['image_1'])).convert("RGB"))
        image_2 = self.clean_transform(Image.open(get_the_image(row['name_2'], row['image_2'])).convert("RGB"))

        return image_1, image_2, row['same_person']

class SiameseDataset(Dataset):
    def __init__(self, df, mode='rgb_denoise'):
        """
        mode: one of {'gray', 'gray_denoise', 'rgb', 'rgb_denoise'}
        """
        self.df = df
        self.mode = mode

        self.gray_transform = transforms.Compose([
            transforms.Resize((105, 105)),
            transforms.ToTensor(),
        ])

        self.gray_noise_transform = transforms.Compose([
            transforms.Resize((105, 105)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.clamp(x + 0.3 * torch.randn_like(x), 0., 1.)),
        ])

        self.rgb_transform = transforms.Compose([
            transforms.Resize((105, 105)),
            transforms.ToTensor(),
        ])

        self.rgb_noise_transform = transforms.Compose([
            transforms.Resize((105, 105)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.clamp(x + 0.3 * torch.randn_like(x), 0., 1.)),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Select transform & image mode
        if self.mode == 'gray':
            transform = self.gray_transform
            convert_mode = "L"
        elif self.mode == 'gray_denoise':
            transform = self.gray_noise_transform
            convert_mode = "L"
        elif self.mode == 'rgb':
            transform = self.rgb_transform
            convert_mode = "RGB"
        elif self.mode == 'rgb_denoise':
            transform = self.rgb_noise_transform
            convert_mode = "RGB"
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        image_1 = transform(Image.open(get_the_image(row['name_1'], row['image_1'])).convert(convert_mode))
        image_2 = transform(Image.open(get_the_image(row['name_2'], row['image_2'])).convert(convert_mode))

        # Return flipped label if using contrastive loss
        label = 1 - row['same_person']  # Comment this out if using BCE
        return image_1, image_2, torch.tensor(label, dtype=torch.float32)