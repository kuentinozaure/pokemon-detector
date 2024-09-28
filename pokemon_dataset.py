from torchvision import transforms
from torch.utils.data import Dataset

class PokemonDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]['image']  # Adjust this according to the dataset structure
        label = self.dataset[idx]['labels']   # Adjust this according to the dataset structure

        if self.transform:
            image = self.transform(image)

        return image, label