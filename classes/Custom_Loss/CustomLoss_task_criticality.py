import torch.nn as nn
import torch

#includes task + criticality loss
class CustomLoss_task_criticality(nn.Module):

    def __init__(self, target_firing_rate=0.02, target_synchrony=1.4,target_branching=1.0,batch_size=25):
        super(CustomLoss_task_criticality, self).__init__()
        self.target_synchrony = torch.tensor([target_synchrony] * batch_size, requires_grad=True)
        self.target_firing_rate = torch.tensor([target_firing_rate] * batch_size,requires_grad=True)
        self.target_criticality = torch.tensor([target_branching] * batch_size,requires_grad=True)

    def forward(self, outputs, targets, criticality, firing_rate, synchrony_fano_factor):
        w_crit = 500
        w_task = 1
        
        task_loss = nn.MSELoss()(outputs.squeeze(), targets)
        criticality_loss = nn.MSELoss()(criticality,self.target_criticality)

        self.task_loss = task_loss
        self.criticality_loss = criticality_loss

        self.rate_loss = torch.tensor([0])
        self.synchrony_loss = torch.tensor([0])

        total_loss = w_task*task_loss + w_crit*criticality_loss
        return total_loss
