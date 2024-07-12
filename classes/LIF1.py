from classes.SpikingNeuron1 import SpikingNeuron1
import torch
import torch.nn as nn

#LIF1 class utilizing SpikingNeuron1
class LIF1(SpikingNeuron1):
    """Parent class for leaky integrate and fire neuron models."""

    def __init__(
        self,
        beta,
        threshold=1.0,
        spike_grad=None,
        surrogate_disable=False,
        init_hidden=False,
        inhibition=False,
        learn_beta=False,
        learn_threshold=False,
        reset_mechanism="subtract",
        state_quant=False,
        output=False,
        graded_spikes_factor=1.0,
        learn_graded_spikes_factor=False,
    ):
        super().__init__(
            threshold,
            spike_grad,
            surrogate_disable,
            init_hidden,
            inhibition,
            learn_threshold,
            reset_mechanism,
            state_quant,
            output,
            graded_spikes_factor,
            learn_graded_spikes_factor,
        )

        self._lif_register_buffer(
            beta,
            learn_beta,
        )
        self._reset_mechanism = reset_mechanism

    def _lif_register_buffer(
        self,
        beta,
        learn_beta,
    ):
        """Set variables as learnable parameters else register them in the
        buffer."""
        self._beta_buffer(beta, learn_beta)

    def _beta_buffer(self, beta, learn_beta):
        if not isinstance(beta, torch.Tensor):
            beta = torch.as_tensor(beta)  # TODO: or .tensor() if no copy
        if learn_beta:
            self.beta = nn.Parameter(beta)
        else:
            self.register_buffer("beta", beta)

    def _V_register_buffer(self, V, learn_V):
        if V is not None:
            if not isinstance(V, torch.Tensor):
                V = torch.as_tensor(V)
        if learn_V:
            self.V = nn.Parameter(V)
        else:
            self.register_buffer("V", V)
