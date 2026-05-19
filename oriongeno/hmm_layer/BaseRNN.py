from typing import Optional, Union, Tuple

import torch
import torch.nn as nn


class BaseRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BaseRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Trainable affine parameters for the input and recurrent paths.
        self.weight_ih = nn.Parameter(torch.randn(hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.randn(hidden_size))
        self.bias_hh = nn.Parameter(torch.randn(hidden_size))
        self.max_num_states = hidden_size

        self.recurrent_init()

    def recurrent_init(self):
        """Initialize recurrent-cell parameters."""
        nn.init.xavier_uniform_(self.weight_ih)
        nn.init.orthogonal_(self.weight_hh)
        nn.init.zeros_(self.bias_ih)
        nn.init.zeros_(self.bias_hh)

    def forward(self, input, hidden):
        """Run one recurrent step.

        Args:
            input: (batch_size, input_size)
            hidden: (batch_size, hidden_size)

        Returns:
            next_hidden: (batch_size, hidden_size)
        """
        igates = torch.matmul(input, self.weight_ih.t()) + self.bias_ih
        hgates = torch.matmul(hidden, self.weight_hh.t()) + self.bias_hh
        next_hidden = torch.tanh(igates + hgates)

        return next_hidden


class BaseHMMCell(nn.Module):
    def __init__(self, n_states, hidden_size):
        super(BaseHMMCell, self).__init__()
        self.n_states = n_states
        self.hidden_size = hidden_size
        self.max_num_states = hidden_size

        self.transition = nn.Parameter(torch.randn(n_states, n_states))
        self.emission = nn.Parameter(torch.randn(n_states, hidden_size))
        self.init = nn.Parameter(torch.randn(n_states))

        self.recurrent_init()

    def recurrent_init(self):
        nn.init.xavier_uniform_(self.transition)
        nn.init.xavier_uniform_(self.emission)
        nn.init.zeros_(self.init)

    def get_initial_state(self, batch_size, **kwargs):
        return torch.zeros(batch_size, self.n_states)

    def emission_probs(self, inputs, **kwargs):
        """Compute emission probabilities for the provided observations.

        Args:
            inputs: (batch_size, n_observation)

        Returns:
            B: (n_states, n_observation)
        """
        B = torch.softmax(self.emission, dim=1)
        emit = torch.matmul(inputs, B)

        return emit

    def forward(self, inputs, states, **kwargs):
        """Run one HMM recurrence step.

        Args:
            inputs: (batch_size, n_observation)
            states: (batch_size, n_states)

        Returns:
            next_states: (batch_size, n_states)
        """
        A = torch.softmax(self.transition, dim=1)
        B = torch.softmax(self.emission, dim=1)
        next_states = torch.matmul(states, A) + torch.matmul(inputs, B.t()) + self.init

        return next_states, next_states


class BaseRNN(nn.Module):
    def __init__(self,
                 cell: nn.Module,
                 batch_first: bool = False,
                 return_sequences: bool = True,
                 return_state: bool = False,
                 reverse: bool = False):
        """
        Recurrent wrapper with configurable sequence and state returns.

        Args:
            cell: The recurrent cell implementation.
            batch_first: If True, inputs are expected as (batch, seq, features);
                otherwise they are expected as (seq, batch, features).
            return_sequences: Whether to return the full output sequence.
            return_state: Whether to return the final hidden state or states.
            reverse: If True, process the sequence in reverse order.
        """
        super(BaseRNN, self).__init__()
        self.cell = cell
        self.batch_first = batch_first
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.reverse = reverse

    def forward(self,
                inputs: torch.Tensor,
                hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Union[
        torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Forward pass for the RNN.

        Args:
            inputs: Input tensor of shape:
                   (batch_size, seq_len, input_size) if batch_first
                   (seq_len, batch_size, input_size) otherwise
            hidden: Initial hidden state tensor or state tuple/list.

        Returns:
            Depending on return_sequences and return_state:
            - If both False: returns last output only
            - If return_sequences True: returns full output sequence
            - If return_state True: returns hidden state(s)
            - If both True: returns (output_sequence, hidden_state(s))
        """
        if self.batch_first:
            # Internal recurrence always iterates over the first dimension.
            inputs = inputs.transpose(0, 1)

        if self.reverse:
            inputs = torch.flip(inputs, [0])

        seq_len, batch_size, _ = inputs.size()

        if hidden is None:
            hidden = self.get_initial_state(inputs, batch_size)


        hx = hidden

        outputs = []
        for t in range(seq_len):
            current_input = inputs[t]
            if isinstance(hx, (tuple, list)) and len(hx) == 2:
                output, hx = self.cell(current_input, hx)
            else:
                hx = self.cell(current_input, hx)
                output = hx
            outputs.append(output)

        if self.return_sequences:
            output_sequence = torch.stack(outputs, dim=0)
            if self.batch_first:
                output_sequence = output_sequence.transpose(0, 1)
        else:
            output_sequence = outputs[-1]
            if self.batch_first:
                output_sequence = output_sequence.unsqueeze(1)

        if not self.return_sequences and not self.return_state:
            return output_sequence
        elif self.return_sequences and not self.return_state:
            return output_sequence
        elif not self.return_sequences and self.return_state:
            return output_sequence, hx
        else:
            return output_sequence, hx

    def get_initial_state(
            self,
            inputs: torch.Tensor,
            batch_size: int) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Return initial hidden states for the wrapped recurrent cell.

        Args:
            inputs: Input tensor used for shape inference when required.
            batch_size: Current batch size.

        Returns:
            Initial hidden state tensor or state tuple.
        """
        device = inputs.device
        if getattr(self.cell, "get_initial_state", None) is not None:
            hidden = self.cell.get_initial_state(inputs=inputs, batch_size=batch_size, device=device)
            return hidden
        else:
            return torch.zeros(batch_size, self.cell.hidden_size, device=device).to(inputs.device)
