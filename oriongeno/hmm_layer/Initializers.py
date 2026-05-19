import torch
import torch.nn as nn
import numpy as np

class EmissionInitializer(nn.Module):
    """Initialize emission probability tensors from a fixed distribution."""

    def __init__(self, dist):
        """Store the distribution that will be broadcast to the requested shape."""
        super(EmissionInitializer, self).__init__()
        self.dist = torch.tensor(dist) if isinstance(dist, np.ndarray) else dist

    def forward(self, shape, dtype=None, device=None):
        """Broadcast the stored distribution across all leading dimensions."""
        assert shape[-1] == self.dist.size(0), (
            f"The last requested dimension must match the distribution size. "
            f"shape={shape}, dist.size={self.dist.size(0)}"
        )

        if dtype is not None:
            dist = self.dist.to(dtype)
        else:
            dist = self.dist

        if device is not None:
            dist = dist.to(device)

        prod_shape = torch.prod(torch.tensor(shape[:-1]))
        tiled_dist = dist.repeat(prod_shape)
        return tiled_dist.view(shape)

    def __repr__(self):
        """Return a concise string representation."""
        return f"EmissionInitializer(dist={self.dist.tolist()})"

    def get_config(self):
        """Return serializable configuration values."""
        return {"dist": self.dist.tolist()}

    @classmethod
    def from_config(cls, config):
        """Create an instance from serialized configuration values."""
        return cls(np.array(config["dist"]))


class ConstantInitializer(torch.nn.Module):
    """Initialize tensors with a fixed scalar or tensor value."""

    def __init__(self, value):
        """Store a scalar or tensor value for later broadcasting."""
        super(ConstantInitializer, self).__init__()
        self.value = torch.tensor(value) if isinstance(value, np.ndarray) else torch.tensor([value]) if np.isscalar(value) else torch.tensor(value)

    def forward(self, shape, dtype=None, device=None):
        """Broadcast the stored value to the requested shape."""
        if dtype is not None:
            value = self.value.to(dtype)
        else:
            value = self.value

        if device is not None:
            value = value.to(device)

        return value.repeat(shape)

    def __repr__(self):
        """Return a concise string representation."""
        if self.value.numel() == 1:
            return f"Const({self.value.item()})"
        elif self.value.ndim == 1:
            return f"Const(size={self.value.size(0)})"
        else:
            return f"Const(shape={self.value.shape})"

    def get_config(self):
        """Return serializable configuration values."""
        return {"value": self.value.tolist() if isinstance(self.value, torch.Tensor) else self.value}

    @classmethod
    def from_config(cls, config):
        """Create an instance from serialized configuration values."""
        return cls(np.array(config["value"]))


def make_gene_pred_emission_kernel(num_labels, smoothing=0.1, num_copies=1, num_models=1, noise_strength=0.001):
    """Create log-emission kernels that favor the matching label for each state."""
    assert smoothing > 0, "Smoothing can not be exactly zero to prevent numerical issues."
    n = num_labels
    probs = np.eye(n)
    probs += -probs * smoothing + (1-probs)*smoothing/(n-1)
    if num_copies > 1:
        probs = np.concatenate([probs[:1]] + [probs[1:]] * num_copies, axis=-2)
    # Replicate the emission matrix once per model.
    probs = np.repeat(probs[np.newaxis, ...], num_models, axis=0)

    return np.log(probs).astype(np.float32)


def make_15_class_emission_kernel(smoothing=0.1, num_copies=1, num_models=1, noise_strength=0.001):
    return make_gene_pred_emission_kernel(
        num_labels=15,
        smoothing=smoothing,
        num_copies=num_copies,
        num_models=num_models,
        noise_strength=noise_strength,
    )


def make_20_class_emission_kernel(smoothing=0.1, num_copies=1, num_models=1, noise_strength=0.001):
    return make_gene_pred_emission_kernel(
        num_labels=20,
        smoothing=smoothing,
        num_copies=num_copies,
        num_models=num_models,
        noise_strength=noise_strength,
    )


background_distribution = np.exp(make_15_class_emission_kernel())
def make_default_emission_init():
    return EmissionInitializer(np.log(background_distribution + 1e-10))


def make_default_insertion_init():
    return ConstantInitializer(np.log(background_distribution + 1e-10))


class EntryInitializer(nn.Module):
    """Initialize profile-HMM entry transition logits."""

    def __init__(self):
        super(EntryInitializer, self).__init__()

    def forward(self, shape, dtype=None, device=None):
        """Give the first entry state all probability mass and spread the rest uniformly."""
        if dtype is None:
            dtype = torch.float32

        p0 = torch.zeros([1] + list(shape[1:]), dtype=dtype, device=device)
        p = torch.log(1 / (shape[0] - 1)) * torch.ones([shape[0] - 1] + list(shape[1:]), dtype=dtype, device=device)

        return torch.cat([p0, p], dim=0)

    def __repr__(self):
        """Return a concise string representation."""
        return "DefaultEntry()"

class ExitInitializer(nn.Module):
    """Initialize profile-HMM exit transition logits."""

    def __init__(self):
        super(ExitInitializer, self).__init__()

    def forward(self, shape, dtype=None, device=None):
        """Initialize exit logits with a uniform half-exit prior."""
        if dtype is None:
            dtype = torch.float32

        return torch.zeros(shape, dtype=dtype, device=device) + torch.log(0.5 / (shape[0] - 1))

    def __repr__(self):
        """Return a concise string representation."""
        return "DefaultExit()"


class MatchTransitionInitializer(nn.Module):
    """Initialize match-state transition logits."""

    def __init__(self, val, i, scale):
        """Select one transition component from a noisy match-transition prior."""
        super(MatchTransitionInitializer, self).__init__()
        self.val = torch.tensor(val)
        self.i = i
        self.scale = scale

    def forward(self, shape, dtype=None, device=None):
        """Sample per-state logits and return the selected transition probability."""
        if dtype is None:
            dtype = torch.float32

        val = self.val.to(dtype).unsqueeze(0)  # [1, len(val)]
        z = torch.normal(mean=0, std=self.scale, size=(shape[0], 1), dtype=dtype, device=device)  # [shape[0], 1]
        val_z = val + z  # [shape[0], len(val)]

        p_exit_desired = 0.5 / (shape[0] - 1)
        prob = torch.softmax(val_z, dim=-1) * (1 - p_exit_desired)  # [shape[0], len(val)]
        return torch.log(prob[:, self.i])  # [shape[0]]

    def __repr__(self):
        """Return a concise string representation."""
        return f"DefaultMatchTransition({self.val[self.i]})"

    def get_config(self):
        """Return serializable configuration values."""
        return {"val": self.val.tolist(), "i": self.i, "scale": self.scale}



class RandomNormalInitializer(nn.Module):
    """Initialize tensors from a normal distribution."""

    def __init__(self, mean=0.0, stddev=0.05):
        super(RandomNormalInitializer, self).__init__()
        self.mean = mean
        self.stddev = stddev

    def forward(self, shape, dtype=None, device=None):
        """Sample a tensor from the configured normal distribution."""
        if dtype is None:
            dtype = torch.float32

        return torch.normal(mean=self.mean, std=self.stddev, size=shape, dtype=dtype, device=device)

    def __repr__(self):
        """Return a concise string representation."""
        return f"Norm({self.mean}, {self.stddev})"

    def get_config(self):
        """Return serializable configuration values."""
        return {"mean": self.mean, "stddev": self.stddev}


def make_default_flank_init():
    return ConstantInitializer(0.)


def make_default_transition_init(MM=1,
                                 MI=-1,
                                 MD=-1,
                                 II=-0.5,
                                 IM=0,
                                 DM=0,
                                 DD=-0.5,
                                 FC=0,
                                 FE=-1,
                                 R=-9,
                                 RF=0,
                                 T=0,
                                 scale=0.1):
    """Create the default transition initializer dictionary."""
    transition_init_kernel = {
        "begin_to_match": EntryInitializer(),
        "match_to_end": ExitInitializer(),
        "match_to_match": MatchTransitionInitializer([MM, MI, MD], 0, scale),
        "match_to_insert": MatchTransitionInitializer([MM, MI, MD], 1, scale),
        "insert_to_match": RandomNormalInitializer(IM, scale),
        "insert_to_insert": RandomNormalInitializer(II, scale),
        "match_to_delete": MatchTransitionInitializer([MM, MI, MD], 2, scale),
        "delete_to_match": RandomNormalInitializer(DM, scale),
        "delete_to_delete": RandomNormalInitializer(DD, scale),
        "left_flank_loop": RandomNormalInitializer(FC, scale),
        "left_flank_exit": RandomNormalInitializer(FE, scale),
        "right_flank_loop": RandomNormalInitializer(FC, scale),
        "right_flank_exit": RandomNormalInitializer(FE, scale),
        "unannotated_segment_loop": RandomNormalInitializer(FC, scale),
        "unannotated_segment_exit": RandomNormalInitializer(FE, scale),
        "end_to_unannotated_segment": RandomNormalInitializer(R, scale),
        "end_to_right_flank": RandomNormalInitializer(RF, scale),
        "end_to_terminal": RandomNormalInitializer(T, scale)
    }
    return transition_init_kernel
