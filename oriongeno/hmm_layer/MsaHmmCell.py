import copy

import torch
import torch.nn as nn

from .Emitter import ProfileHMMEmitter
from .Transitioner import ProfileHMMTransitioner
from .Utility import get_num_states

class HmmCell(nn.Module):
    """Generic HMM cell that performs one forward or backward recursion step."""
    def __init__(self, num_states, dim, emitter, transitioner, use_step_counter=False, use_fake_step_counter=False, device=None, **kwargs):
        super(HmmCell, self).__init__(**kwargs)
        self.num_states = num_states
        self.num_models = len(self.num_states)
        self.max_num_states = max(self.num_states)
        self.dim = dim
        emitter = [emitter] if not isinstance(emitter, list) else emitter
        self.emitter = nn.ModuleList(emitter)
        self.transitioner = transitioner
        self.epsilon = 1e-16
        self.reverse = False
        self.use_step_counter = use_step_counter
        self.use_fake_step_counter = use_fake_step_counter

        self.recurrent_init()

    def recurrent_init(self):
        """Prepare cached tensors used by recurrent evaluation."""
        self.transitioner.recurrent_init()
        for em in self.emitter:
            em.recurrent_init()
        self.log_A_dense = self.transitioner.make_log_A()
        self.log_A_dense_t = torch.transpose(self.log_A_dense, 1, 2)
        self.init_dist = self.make_initial_distribution()
        if not self.reverse and self.use_step_counter:
            self.step_counter = torch.tensor(-1, dtype=torch.int32)

    def make_initial_distribution(self):
        """Return the transitioner's initial state distribution."""
        return self.transitioner.make_initial_distribution()

    def _runtime_dtype_device(self, inputs=None, device=None):
        """Return dtype/device for tensors created during HMM inference."""
        if torch.is_tensor(inputs):
            dtype = inputs.dtype
            resolved_device = inputs.device
        else:
            try:
                parameter = next(self.parameters())
                dtype = parameter.dtype
                resolved_device = parameter.device
            except StopIteration:
                dtype = torch.float32
                resolved_device = torch.device("cpu")
        return dtype, device or resolved_device

    def emission_probs(self, inputs, end_hints=None, training=False):
        """Compute state emission probabilities for one observation."""
        em_probs = self.emitter[0](inputs, end_hints=end_hints, training=training)
        for em in self.emitter[1:]:
            em_probs *= em(inputs, end_hints=end_hints, training=training)
        return em_probs

    def forward(self, emission_probs, states, training=None, init=False):
        """Advance scaled forward probabilities by one observation."""
        old_scaled_forward, old_loglik = states
        old_scaled_forward = old_scaled_forward.view(self.num_models, -1, self.max_num_states)
        if init:
            R = old_scaled_forward
        else:
            R = self.transitioner(old_scaled_forward)
        E = emission_probs.view(self.num_models, -1, self.max_num_states)

        q = R.shape[1] // E.shape[1]
        R = R.view(self.num_models, -1, q, self.max_num_states)
        E = E.view(self.num_models, -1, 1, self.max_num_states)
        old_loglik = old_loglik.view(self.num_models, -1, q, 1)
        epsilon = torch.as_tensor(self.epsilon, dtype=E.dtype, device=E.device)
        E = torch.maximum(E, epsilon)
        R = torch.maximum(R, epsilon)
        scaled_forward = E * R
        S = torch.sum(scaled_forward, dim=-1, keepdim=True)
        loglik = old_loglik + torch.log(S)
        scaled_forward /= S
        scaled_forward = scaled_forward.view(-1, q * self.max_num_states)
        loglik = loglik.view(-1, q)
        new_state = [scaled_forward, loglik]
        if self.reverse:
            output = torch.log(R)
            output = output.view(-1, q * self.max_num_states)
            old_loglik = old_loglik.view(-1, q)
            output = torch.cat([output, old_loglik], dim=-1)
        else:
            output = torch.log(scaled_forward)
            output = torch.cat([output, loglik], dim=-1)
        if not self.reverse and self.use_step_counter:
            self.step_counter += 1
        return output, new_state

    def get_initial_state(self, inputs=None, batch_size=None, parallel_factor=1, device=None):
        """Return recurrent state tensors for serial or chunk-parallel evaluation."""
        dtype, device = self._runtime_dtype_device(inputs=inputs, device=device)
        if parallel_factor == 1:
            if self.reverse:
                init_dist = torch.ones((self.num_models * batch_size, self.max_num_states), dtype=dtype, device=device)
            else:
                init_dist = (
                    self.make_initial_distribution()
                    .repeat(batch_size, 1, 1)
                    .transpose(0, 1)
                    .view(-1, self.max_num_states)
                    .to(device=device, dtype=dtype)
                )
            loglik = torch.zeros((self.num_models * batch_size, 1), dtype=dtype, device=device)
            return [init_dist.to(device=device, dtype=dtype), loglik.to(device=device, dtype=dtype)]
        else:
            indices = torch.arange(self.max_num_states, device=device).repeat(self.num_models * batch_size)
            init_dist = torch.nn.functional.one_hot(indices, num_classes=self.max_num_states).to(dtype=dtype, device=device)
            if self.reverse:
                init_dist_chunk = init_dist.clone().view(self.num_models * batch_size, self.max_num_states, self.max_num_states)
                first_emissions = inputs[:, 0, :].view(self.num_models, batch_size // parallel_factor, parallel_factor, self.max_num_states)
                first_emissions = torch.roll(first_emissions, shifts=-1, dims=2).view(self.num_models * batch_size, 1, self.max_num_states)
                init_dist_chunk *= first_emissions
            else:
                init_dist_chunk = init_dist
            init_dist_chunk = init_dist_chunk.view(self.num_models, batch_size * self.max_num_states, self.max_num_states).to(device=device, dtype=dtype)
            init_dist_trans = self.transitioner(init_dist_chunk).view(self.num_models, batch_size // parallel_factor, parallel_factor, self.max_num_states * self.max_num_states)
            is_first_chunk = torch.zeros((self.num_models, batch_size // parallel_factor, parallel_factor - 1, self.max_num_states * self.max_num_states), dtype=dtype, device=device)
            if self.reverse:
                is_first_chunk = torch.cat([is_first_chunk, torch.ones_like(is_first_chunk[..., :1, :])], dim=2).to(device=device, dtype=dtype)
            else:
                is_first_chunk = torch.cat([torch.ones_like(is_first_chunk[..., :1, :]), is_first_chunk], dim=2).to(device=device, dtype=dtype)
            init_dist = init_dist.view(self.num_models, batch_size // parallel_factor, parallel_factor, self.max_num_states * self.max_num_states)
            init_dist = is_first_chunk * init_dist + (1 - is_first_chunk) * init_dist_trans
            init_dist = init_dist.view(self.num_models * batch_size, self.max_num_states * self.max_num_states)
            loglik = torch.zeros((self.num_models * batch_size, self.max_num_states), dtype=dtype, device=device)
            return [init_dist.to(device=device, dtype=dtype), loglik.to(device=device, dtype=dtype)]

    def get_aux_loss(self):
        return sum([em.get_aux_loss() for em in self.emitter])

    def get_prior_log_density(self):
        em_priors = [torch.sum(em.get_prior_log_density(), dim=1) for em in self.emitter]
        trans_priors = self.transitioner.get_prior_log_densities()
        prior = sum(em_priors) + sum(trans_priors.values())
        return prior

    def make_reverse_direction_offspring(self):
        """Create a deep-copied cell configured for reverse-direction recursion."""
        reverse_cell = copy.deepcopy(HmmCell(self.num_states, self.dim, self.emitter, self.transitioner))
        reverse_cell.reverse = True
        reverse_cell.transitioner.reverse = True
        reverse_cell.recurrent_init()
        return reverse_cell

    def reverse_direction(self, reverse=True):
        self.reverse = reverse
        self.transitioner.reverse = reverse

class MsaHmmCell(HmmCell):
    """Profile-HMM cell with default emitter and transitioner wiring."""
    def __init__(self, length, dim=24, emitter=None, transitioner=None, **kwargs):
        if emitter is None:
            emitter = ProfileHMMEmitter()
        if transitioner is None:
            transitioner = ProfileHMMTransitioner()
        self.length = [length] if not isinstance(length, list) else length
        super(MsaHmmCell, self).__init__(get_num_states(self.length), dim, emitter, transitioner, **kwargs)
        for em in self.emitter:
            em.set_lengths(self.length)
        self.transitioner.set_lengths(self.length)
