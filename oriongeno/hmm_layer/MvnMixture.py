import torch
import torch.nn as nn
import math

from .Utility import DefaultDiagBijector, FillScaleTriL


class MvnMixture(nn.Module):
    """Multivariate normal mixture used by HMM emission components."""

    def __init__(self,
                 dim,
                 kernel,
                 mixture_coeff_kernel=None,
                 diag_only=True,
                 diag_bijector=DefaultDiagBijector(1.),
                 precomputed=False,
                 **kwargs):
        """Initialize the object."""
        super(MvnMixture, self).__init__(**kwargs)
        self.dim = dim
        self.kernel = torch.tensor(kernel, dtype=torch.float32)
        if mixture_coeff_kernel is not None:
            self.mixture_coeff_kernel = torch.tensor(mixture_coeff_kernel, dtype=torch.float32)
        else:
            self.mixture_coeff_kernel = None
        self.num_components = self.kernel.shape[2]
        self.diag_only = diag_only
        self.diag_bijector = diag_bijector
        self.precomputed = precomputed
        self.scale_tril = FillScaleTriL(diag_bijector=diag_bijector)
        self.constant = self.dim * math.log(2 * math.pi)
        self.scale = None
        self.pinv = None

        assert len(self.kernel.shape) == 4
        if diag_only:
            assert self.kernel.shape[-1] == 2 * dim
        else:
            assert self.kernel.shape[-1] == dim + dim * (dim + 1) // 2
        if self.mixture_coeff_kernel is not None:
            assert len(self.mixture_coeff_kernel.shape) == 3
            assert self.mixture_coeff_kernel.shape == self.kernel.shape[:3]
        else:
            assert self.num_components == 1

    def component_expectations(self):
        """Return the mean vector for each mixture component."""
        mu = self.kernel[..., :self.dim]
        return mu

    def expectation(self):
        """Compute the expectation of the mixture distribution."""
        if self.num_components == 1:
            return self.component_expectations()[..., 0, :]
        else:
            comp_exp = self.component_expectations()
            mix_coeff = self.mixture_coefficients()
            return torch.sum(comp_exp * mix_coeff.unsqueeze(-1), -2)

    def component_scales(self, return_scale_diag=False, return_inverse=False):
        """Return component scale matrices or diagonal scales."""
        if not self.precomputed or self.scale is None:
            if self.diag_only:
                scale_diag = self.diag_bijector.forward(self.kernel[..., self.dim:])
                scale_diag += 1e-8
                scale = scale_diag if return_scale_diag else torch.eye(self.dim, device=self.kernel.device).unsqueeze(
                    0).unsqueeze(0).unsqueeze(0) * scale_diag.unsqueeze(-1)
                if return_inverse:
                    pinv = 1. / scale_diag
            else:
                scale_kernel = self.kernel[..., self.dim:]
                scale = self.scale_tril.forward(scale_kernel)
                if return_inverse:
                    pinv = torch.linalg.pinv(scale)
                if return_scale_diag:
                    scale = torch.diagonal(scale, dim1=-2, dim2=-1)
        return (scale, pinv) if return_inverse else scale

    def component_covariances(self):
        """Return component covariance matrices or diagonal variances."""
        scale = self.component_scales(return_scale_diag=self.diag_only)
        if self.diag_only:
            return torch.square(scale)
        else:
            return torch.matmul(scale, scale.transpose(-1, -2))

    def component_log_pdf(self, inputs):
        """Compute per-component log probability density."""
        mu = self.component_expectations()
        scale_diag, pinv = self.component_scales(return_scale_diag=True, return_inverse=True)
        log_det = 2 * torch.sum(torch.log(scale_diag), -1)  # (k1, k2, c, 1)
        diff = inputs.unsqueeze(1).unsqueeze(2) - mu.unsqueeze(-2)  # (k1, k2, c, b, d)
        if self.diag_only:
            pinv_sq = torch.square(pinv)  # (k1, k2, c, d)
            diff_sq = torch.square(diff)
            MD_sq_components = torch.sum(diff_sq * pinv_sq.unsqueeze(-2), -1)  # (k1, k2, c, b)
        else:
            y = torch.matmul(diff, pinv.transpose(-1, -2))  # (k1, k2, c, b, d)
            MD_sq_components = torch.sum(torch.square(y), -1)
        MD_sq_components = MD_sq_components.transpose(1, 3)
        log_pdf = -0.5 * (self.constant + log_det.unsqueeze(1) + MD_sq_components)
        return log_pdf

    def mixture_coefficients(self):
        """Compute normalized mixture coefficients."""
        return torch.softmax(self.mixture_coeff_kernel, dim=-1)

    def log_pdf(self, inputs):
        """Compute mixture log probability density."""
        log_pdf_components = self.component_log_pdf(inputs)
        if self.num_components == 1:
            return log_pdf_components[..., 0]
        else:
            return torch.logsumexp(log_pdf_components + torch.log(self.mixture_coefficients().unsqueeze(1)), -1)

    def get_regularization_L2_loss(self):
        """Return an L2 penalty over scale parameters."""
        return torch.mean(torch.sum(torch.square(self.kernel[..., self.dim:]), dim=-1))
