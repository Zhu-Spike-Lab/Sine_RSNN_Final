from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd

# L3: Changing Amplitude (20-80), Changing Period (40-100)
class SineWaveDataset3(Dataset):
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

        # L3: without explicit time
        amplitude_vector = torch.tensor([amplitude] * self.num_timesteps, dtype=torch.float32)
        period_vector = torch.tensor([period] * self.num_timesteps, dtype=torch.float32)
        
        sine_wave_vector = torch.tensor(sine_wave, dtype=torch.float32)
        input_vector = torch.stack([amplitude_vector,period_vector],dim=1)  # Shape: [num_timesteps, 2]
        target_vector = torch.stack([sine_wave_vector],dim=1)
        return input_vector, sine_wave_vector
