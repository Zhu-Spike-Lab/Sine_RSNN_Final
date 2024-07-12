from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd

# L400: Changing Amplitude (20-80), Changing Period (40-100), Clock-like Input  Limited time input
class SineWaveDataset400(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.num_timesteps = 300
        #timesteps manually encoded in target_output

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        amplitude = self.data.iloc[idx, 0]
        period = self.data.iloc[idx, 1]
        sine_wave = eval(self.data.iloc[idx, 2])  # assuming the sine_wave is stored as a string representation of a list
        
        # L4: A & P for the first 10ms
        amplitude_vector = torch.tensor([amplitude] * 10 + [0] * (self.num_timesteps-10), dtype=torch.float32)
        time_vector = torch.tensor([i for i in range(period)] * (self.num_timesteps//int(period)+1), dtype=torch.float32)
        time_vector = time_vector[:self.num_timesteps]
        period_vector = torch.tensor([period] * 10 + [0] * (self.num_timesteps-10), dtype=torch.float32)

        sine_wave_vector = torch.tensor(sine_wave, dtype=torch.float32)
        input_vector = torch.stack([amplitude_vector,period_vector,time_vector],dim=1)  # Shape: [num_timesteps, 2]
        target_vector = torch.stack([sine_wave_vector],dim=1)
        return input_vector, sine_wave_vector
