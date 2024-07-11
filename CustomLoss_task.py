import torch.nn as nn
import torch

class CustomLoss_task(nn.Module):
    def __init__(self, target_firing_rate=0.02,target_synchrony=1.4, target_branching=1.0,batch_size=25):
        super(CustomLoss_task, self).__init__()
        self.target_synchrony = torch.tensor([target_synchrony] * batch_size, requires_grad=True)
        self.target_firing_rate = torch.tensor([target_firing_rate] * batch_size,requires_grad=True)
        self.target_branching = torch.tensor([target_branching] * batch_size,requires_grad=True)

    def forward(self, outputs, targets, firing_rate, synchrony_fano_factor, criticality):
        w_task = 1
        task_loss = nn.MSELoss()(outputs.squeeze(), targets)
        self.task_loss = task_loss

        self.rate_loss = torch.tensor([0])
        self.criticality_loss = torch.tensor([0])
        self.synchrony_loss = torch.tensor([0])

        total_loss = w_task*task_loss 
        return total_loss
