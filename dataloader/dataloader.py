import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
import os
import numpy as np
from .augmentations import DataTransform, DataTransform_FD
import torch.fft as fft

class Load_Dataset(Dataset):
    def __init__(self, dataset, dataset_configs):
        super().__init__()
        self.num_channels = dataset_configs.input_channels

        # Load samples
        x_data = dataset["samples"]

        # Check samples dimensions.
        # The dimension of the data is expected to be (N, C, L)
        # where N is the #samples, C: #channels, and L is the sequence length
        if len(x_data.shape) == 2:
            x_data = x_data.unsqueeze(1)
        elif len(x_data.shape) == 3 and x_data.shape[1] != self.num_channels:
            x_data = x_data.transpose(0, 2, 1)

        x_data_weak, x_data_strong, x_data_strong2 = DataTransform(x_data, dataset_configs)
        
        # Convert to torch tensor
        if isinstance(x_data, np.ndarray):
            x_data = torch.from_numpy(x_data)

        if isinstance(x_data_weak, np.ndarray):
            x_data_weak = torch.from_numpy(x_data_weak)

        if isinstance(x_data_strong, np.ndarray):
            x_data_strong = torch.from_numpy(x_data_strong)

        if isinstance(x_data_strong2, np.ndarray):
            x_data_strong2 = torch.from_numpy(x_data_strong2)

        # Load labels
        y_data = dataset.get("labels")
        if y_data is not None and isinstance(y_data, np.ndarray):
            y_data = torch.from_numpy(y_data)

        """Transfer x_data to Frequency Domain. If use fft.fft, the output has the same shape; if use fft.rfft, 
        the output shape is half of the time window."""
        window_length = x_data.shape[-1]
        x_data_f = fft.fft(x_data).abs() 
        x_data_f_weak, x_data_f_strong, x_data_f_strong2 = DataTransform_FD(x_data_f, dataset_configs)

        # Convert to torch tensor
        if isinstance(x_data_f, np.ndarray):
            x_data_f = torch.from_numpy(x_data_f)

        if isinstance(x_data_f_weak, np.ndarray):
            x_data_f_weak = torch.from_numpy(x_data_f_weak)

        if isinstance(x_data_f_strong, np.ndarray):
            x_data_f_strong = torch.from_numpy(x_data_f_strong)

        if isinstance(x_data_f_strong2, np.ndarray):
            x_data_f_strong2 = torch.from_numpy(x_data_f_strong2)


        # Normalize data
        if dataset_configs.normalize:
            data_mean4 = torch.mean(x_data, dim=(0, 2))
            data_std4 = torch.std(x_data, dim=(0, 2))
            self.transform4 = transforms.Normalize(mean=data_mean4, std=data_std4)
            data_mean = torch.mean(x_data_weak, dim=(0, 2))
            data_std = torch.std(x_data_weak, dim=(0, 2))
            self.transform = transforms.Normalize(mean=data_mean, std=data_std)
            data_mean2 = torch.mean(x_data_strong, dim=(0, 2))
            data_std2 = torch.std(x_data_strong, dim=(0, 2))
            self.transform2 = transforms.Normalize(mean=data_mean2, std=data_std2)
            data_mean3 = torch.mean(x_data_strong2, dim=(0, 2))
            data_std3 = torch.std(x_data_strong2, dim=(0, 2))
            self.transform3 = transforms.Normalize(mean=data_mean3, std=data_std3)

            data_mean5 = torch.mean(x_data_f, dim=(0, 2))
            data_std5 = torch.std(x_data_f, dim=(0, 2))
            self.transform5 = transforms.Normalize(mean=data_mean5, std=data_std5)
            data_mean6 = torch.mean(x_data_f_weak, dim=(0, 2))
            data_std6 = torch.std(x_data_f_weak, dim=(0, 2))
            self.transform6 = transforms.Normalize(mean=data_mean6, std=data_std6)
            data_mean7 = torch.mean(x_data_f_strong, dim=(0, 2))
            data_std7 = torch.std(x_data_f_strong, dim=(0, 2))
            self.transform7 = transforms.Normalize(mean=data_mean7, std=data_std7)
            data_mean8 = torch.mean(x_data_f_strong2, dim=(0, 2))
            data_std8 = torch.std(x_data_f_strong2, dim=(0, 2))
            self.transform8 = transforms.Normalize(mean=data_mean8, std=data_std8)

        self.x_data = x_data.float()
        self.x_data_weak = x_data_weak.float()
        self.x_data_strong = x_data_strong.float()
        self.x_data_strong2 = x_data_strong2.float()
        self.x_data_f = x_data_f.float()
        self.x_data_f_weak = x_data_f_weak.float()
        self.x_data_f_strong = x_data_f_strong.float()
        self.x_data_f_strong2 = x_data_f_strong2.float()
        self.y_data = y_data.long() if y_data is not None else None
        self.len = x_data.shape[0]

    def __getitem__(self, index):
        x = self.x_data[index]
        x_weak = self.x_data_weak[index]
        x_strong = self.x_data_strong[index]
        x_strong2 = self.x_data_strong2[index]
        x_f = self.x_data_f[index]
        x_f_weak = self.x_data_f_weak[index]
        x_f_strong = self.x_data_f_strong[index]
        x_f_strong2 = self.x_data_f_strong2[index]
        if self.transform4:
            x = self.transform4(self.x_data[index].reshape(self.num_channels, -1, 1)).reshape(self.x_data[index].shape)

        if self.transform:
            x_weak = self.transform(self.x_data_weak[index].reshape(self.num_channels, -1, 1)).reshape(self.x_data_weak[index].shape)
        if self.transform2:    
            x_strong = self.transform2(self.x_data_strong[index].reshape(self.num_channels, -1, 1)).reshape(self.x_data_strong[index].shape)
        if self.transform3:    
            x_strong2 = self.transform3(self.x_data_strong2[index].reshape(self.num_channels, -1, 1)).reshape(self.x_data_strong2[index].shape)
        y = self.y_data[index] if self.y_data is not None else None
        return x_weak, y, index, x_strong, x_strong2, x, x_f_weak, x_f_strong, x_f_strong2, x_f

    def __len__(self):
        return self.len


def data_generator(data_path, domain_id, dataset_configs, hparams, dtype):
    # loading dataset file from path
    dataset_file = torch.load(os.path.join(data_path, f"{dtype}_{domain_id}.pt"))

    # Loading datasets
    dataset = Load_Dataset(dataset_file, dataset_configs)
    dataset_length = len(dataset)

    if dtype == "test":  # you don't need to shuffle or drop last batch while testing
        shuffle = False
        drop_last = False
    else:
        shuffle = dataset_configs.shuffle
        drop_last = dataset_configs.drop_last

    # Dataloaders
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=hparams["batch_size"],
                                              shuffle=shuffle,
                                              drop_last=drop_last,
                                              num_workers=0)

    return data_loader, dataset_length
