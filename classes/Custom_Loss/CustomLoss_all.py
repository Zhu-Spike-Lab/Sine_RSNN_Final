import torch.nn as nn
import torch

#includes task + criticality + synchrony + rate loss
class CustomLoss_all(nn.Module):

    def __init__(self, target_firing_rate=0.02, target_synchrony=1.4, target_branching=1.0,batch_size=25):
        super(CustomLoss_all, self).__init__()
        self.target_synchrony = torch.tensor([target_synchrony] * batch_size, requires_grad=True)
        self.target_firing_rate = torch.tensor([target_firing_rate] * batch_size,requires_grad=True)
        self.target_branching = torch.tensor([target_branching] * batch_size,requires_grad=True)

    def forward(self, outputs, targets, criticality, firing_rate, synchrony_fano_factor):

        w_crit = 500
        w_rate = 2000
        w_sync = 1000
        w_task = 1
        
        task_loss = nn.MSELoss()(outputs.squeeze(), targets)
        rate_loss = nn.MSELoss()(firing_rate, self.target_firing_rate)
        criticality_loss = nn.MSELoss()(criticality,self.target_branching)
        synchrony_loss = nn.MSELoss()(synchrony_fano_factor,self.target_synchrony)

        self.task_loss = task_loss
        self.rate_loss = rate_loss
        self.criticality_loss = criticality_loss
        self.synchrony_loss = synchrony_loss

        total_loss = w_task*task_loss + w_rate*rate_loss + w_crit*criticality_loss + w_sync*synchrony_loss
        return total_loss
