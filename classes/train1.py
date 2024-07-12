import torch
from classes.helper1 import simple_branching_param, fano_factor, count_spikes
import numpy as np

#function to train and save the variables to npz files!
def train_model(args):
    model_idx, model, optimizer, dataloader,step_dataloader, criterion, criterium_idx, num_epochs, num_timesteps= args
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0

        for i, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs, targets

            optimizer.zero_grad()

            outputs = torch.empty(0, dtype=torch.float32, requires_grad=True)
            firing_rate_per_batch = torch.empty(0, dtype=torch.float32, requires_grad=True)
            criticality_per_batch = torch.empty(0, dtype=torch.float32, requires_grad=True)
            synchrony_per_batch = torch.empty(0, dtype=torch.float32, requires_grad=True)

            for input, target in zip(inputs, targets):
                output, spikes = model(input)
                spikes = spikes.T
                outputs = torch.cat((outputs, output.view(1, -1)))

                firing_rate = count_spikes(spikes) / 30000
                firing_rate_per_batch = torch.cat((firing_rate_per_batch, firing_rate))

                criticality = simple_branching_param(1, spikes).reshape([1])
                criticality_per_batch = torch.cat((criticality_per_batch, criticality))

                synchrony_fano_factor = fano_factor(num_timesteps, spikes).reshape([1])
                synchrony_per_batch = torch.cat((synchrony_per_batch, synchrony_fano_factor))

            loss = criterion(outputs, targets, criticality_per_batch, firing_rate_per_batch, synchrony_per_batch)
            print("loss:", loss)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if epoch % 5 == 0:
                np.savez(f'dataMP/level{step_dataloader}_loss{criterium_idx}_epoch{epoch}_batch{i}.npz',
                         task_loss=criterion.task_loss.item(),
                         criticality_loss = criterion.criticality_loss.item(),
                         synchrony_loss = criterion.synchrony_loss.item(),
                         spikes=spikes.detach().numpy(),
                         input_weights=model.l1.weight.data.detach().numpy(),
                         rec_weights=model.rlif1.recurrent.weight.data.detach().numpy(),
                         output_weights=model.l2.weight.data.detach().numpy(),
                         inputs=inputs.detach().numpy(),
                         outputs=outputs.detach().numpy(),
                         targets=targets.detach().numpy())
