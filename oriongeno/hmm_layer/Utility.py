import math

import numpy as np
import torch
import torch.nn.functional as F


def get_num_states(lengths):
    """Return the number of implicit profile-HMM states for each model length."""
    return [2 * l + 3 for l in lengths]


def get_num_states_implicit(lengths):
    """Return the number of explicit profile-HMM states, including silent states."""
    return [3 * l + 5 for l in lengths]


def inverse_softplus(features):
    features = torch.tensor(features) if not isinstance(features, torch.Tensor) else features
    result = torch.log(torch.expm1(features))
    return result.to(features.dtype)


class DefaultDiagBijector:
    def __init__(self, base_variance, epsilon=1e-05):
        base_std = np.sqrt(base_variance).astype(np.float32)
        self.scale_diag_init = inverse_softplus(torch.tensor(base_std))
        self.epsilon = epsilon

    def forward(self, x):
        return F.softplus(x + self.scale_diag_init) + self.epsilon

    def inverse(self, y):
        return inverse_softplus(y - self.epsilon) - self.scale_diag_init


def fill_triangular(x, upper=False):
    """Convert a flattened triangular vector into a square triangular matrix."""
    x = torch.as_tensor(x)

    m = x.shape[-1]
    n = int((math.sqrt(8 * m + 1) - 1) / 2)
    if n * (n + 1) // 2 != m:
        raise ValueError(
            f"The rightmost dimension ({m}) cannot represent a triangular matrix."
        )

    output_shape = list(x.shape[:-1]) + [n, n]
    matrix = torch.zeros(output_shape, dtype=x.dtype, device=x.device)

    if upper:
        row_indices, col_indices = torch.triu_indices(n, n)
        matrix[..., row_indices, col_indices] = x
    else:
        row_indices, col_indices = torch.tril_indices(n, n)
        matrix[..., row_indices, col_indices] = x

    return matrix


def fill_triangular_inverse(x, upper=False):
    """Flatten the selected triangular part of a square matrix."""
    n = x.shape[-1]
    m = (n * (n + 1)) // 2
    ndims = len(x.shape)

    if upper:
        initial_elements = x[..., 0, :]
        triangular_portion = x[..., 1:, :]
    else:
        initial_elements = torch.flip(x[..., -1, :], dims=[ndims - 2])
        triangular_portion = x[..., :-1, :]

    rotated_triangular_portion = torch.flip(
        torch.flip(triangular_portion, dims=[ndims - 1]),
        dims=[ndims - 2],
    )
    consolidated_matrix = triangular_portion + rotated_triangular_portion

    end_sequence = torch.view(consolidated_matrix, x.shape[:-2] + (n * (n - 1),))
    y = torch.cat([initial_elements, end_sequence[..., :m - n]], dim=-1)

    return y


class FillScaleTriL(torch.nn.Module):
    def __init__(self, diag_bijector):
        super(FillScaleTriL, self).__init__()
        self.diag_bijector = diag_bijector

    def forward(self, x):
        y = fill_triangular(x)
        diag = torch.diagonal(y, dim1=-2, dim2=-1)
        transformed_diag = self.diag_bijector.forward(diag)
        y = y.clone()
        y.diagonal(dim1=-2, dim2=-1).copy_(transformed_diag)
        return y

    def inverse(self, y):
        diag = torch.diagonal(y, dim1=-2, dim2=-1)
        transformed_diag = self.diag_bijector.inverse(diag)
        y = y.clone()
        y.diagonal(dim1=-2, dim2=-1).copy_(transformed_diag)
        x = fill_triangular_inverse(y)
        return x
