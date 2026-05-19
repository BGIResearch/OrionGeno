import torch
import torch.nn as nn
from typing import Optional, Tuple, Union

from .utils import show_value


class Bidirectional(nn.Module):
    """Simple bidirectional wrapper for forward and backward RNNs with shared state.

    Args:
        layer: Forward `nn.Module` instance.
        merge_mode: Mode by which outputs of the forward and backward RNNs
            will be combined. One of `{"sum", "concat"}`.
            If `None`, the outputs will not be combined,
            they will be returned as a list. Defaults to `"concat"`.
        backward_layer: Backward `nn.Module` instance.
    """

    def __init__(
            self,
            layer: nn.Module,
            backward_layer: nn.Module,
            merge_mode: str = "concat",
    ):
        super().__init__()
        if not isinstance(layer, nn.Module):
            raise ValueError(
                "Please initialize `Bidirectional` layer with a "
                f"`nn.Module` instance. Received: {layer}"
            )
        if backward_layer is not None and not isinstance(backward_layer, nn.Module):
            raise ValueError(
                "`backward_layer` need to be a `nn.Module` "
                f"instance. Received: {backward_layer}"
            )
        if merge_mode not in ["sum", "concat", None]:
            raise ValueError(
                f"Invalid merge mode. Received: {merge_mode}. "
                "Merge mode should be one of "
                '{"sum", "concat", None}'
            )

        self.forward_layer = layer
        self.backward_layer = backward_layer
        self._verify_layer_config()

        self.merge_mode = merge_mode
        self.return_sequences = hasattr(layer, "return_sequences") and layer.return_sequences
        self.return_state = hasattr(layer, "return_state") and layer.return_state

    def _create_backward_layer(self, layer: nn.Module) -> nn.Module:
        """Create a backward layer with the same configuration as the forward layer."""
        if isinstance(layer, nn.GRU):
            backward_layer = nn.GRU(
                input_size=layer.input_size,
                hidden_size=layer.hidden_size,
                num_layers=layer.num_layers,
                bias=layer.bias,
                batch_first=layer.batch_first,
                dropout=layer.dropout,
                bidirectional=False,
            )
        elif isinstance(layer, nn.RNN):
            backward_layer = nn.RNN(
                input_size=layer.input_size,
                hidden_size=layer.hidden_size,
                num_layers=layer.num_layers,
                bias=layer.bias,
                batch_first=layer.batch_first,
                dropout=layer.dropout,
                nonlinearity=layer.nonlinearity,
                bidirectional=False,
            )
        else:
            raise ValueError(f"Unsupported layer type: {type(layer)}")
        return backward_layer

    def _verify_layer_config(self):
        """Ensure the forward and backward layers have valid common properties."""
        common_attributes = ("batch_first", "hidden_size")
        for a in common_attributes:
            forward_value = getattr(self.forward_layer, a, None)
            backward_value = getattr(self.backward_layer, a, None)
            if forward_value != backward_value:
                raise ValueError(
                    "Forward layer and backward layer are expected to have "
                    f'the same value for attribute "{a}", got '
                    f'"{forward_value}" for forward layer and '
                    f'"{backward_value}" for backward layer'
                )

    def forward(
            self,
            sequences: torch.Tensor,
            initial_state: Optional[Tuple[torch.Tensor]] = None,
            **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple[torch.Tensor]]]:
        """Forward pass of the bidirectional layer.

        Args:
            sequences: Input tensor of shape (batch_size, seq_len, input_size) if batch_first=True,
                       else (seq_len, batch_size, input_size).
            initial_state: Initial states for the forward and backward layers. If None, defaults to zeros.

        Returns:
            If return_state is False, returns the merged output tensor.
            If return_state is True, returns a tuple of (output, (forward_states, backward_states)).
        """
        if initial_state is not None:
            half = len(initial_state) // 2
            forward_state = initial_state[:half]
            backward_state = initial_state[half:]
        else:
            forward_state, backward_state = None, None

        # Run the two directional recurrences with the caller-provided states.
        show_value(forward_state[0], "03.forward_layer_state_old_scaled_forward")
        show_value(forward_state[1], "03.forward_layer_state_old_loglike")
        show_value(backward_state[0], "03.backward_layer_old_state_scaled_forward")
        show_value(backward_state[1], "03.backward_layer_old_state_loglike")
        forward_output, forward_states = self.forward_layer(sequences, forward_state)
        backward_output, backward_states = self.backward_layer(sequences, backward_state)

        show_value(forward_states[0], "03.forward_layer_state_new_scaled_forward")
        show_value(forward_states[1], "03.forward_layer_state_new_loglike")
        show_value(backward_states[0], "03.backward_layer_new_state_scaled_forward")
        show_value(backward_states[1], "03.backward_layer_new_state_loglike")

        show_value(forward_output, "03.forward_layer_output")
        show_value(backward_output, "04.backward_layer_output")

        if self.return_sequences:
            backward_output = torch.flip(backward_output, dims=[1])
        # Merge outputs after aligning the backward sequence to forward coordinates.
        if self.merge_mode == "concat":
            output = torch.cat([forward_output, backward_output], dim=-1)
        elif self.merge_mode == "sum":
            output = forward_output + backward_output
        elif self.merge_mode is None:
            output = [forward_output, backward_output]
        else:
            raise ValueError(f"Unsupported merge_mode: {self.merge_mode}")

        if self.return_state:
            return output, *forward_states, *backward_states
        return output

    def reset_states(self):
        """Reset the hidden states of the forward and backward layers."""
        if isinstance(self.forward_layer, (nn.RNN, nn.GRU)):
            self.forward_layer.reset_parameters()
        if isinstance(self.backward_layer, (nn.RNN, nn.GRU)):
            self.backward_layer.reset_parameters()

    @property
    def states(self) -> Optional[Tuple[torch.Tensor]]:
        """Return the current states of the forward and backward layers."""
        forward_states = getattr(self.forward_layer, "hidden_state", None)
        backward_states = getattr(self.backward_layer, "hidden_state", None)
        if forward_states is not None and backward_states is not None:
            return (forward_states, backward_states)
        return None
