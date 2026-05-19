import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .Transitioner import make_transition_matrix_from_indices
from .gene_pred_hmm_topology import (
    BASE_GENE_PRED_STATE_TO_INDEX,
    GENE_PRED_BASE_LABEL_DIM,
    expanded_num_states,
    expanded_transition_edges,
)
from .utils import show_value


class GenePredMultiHMMTransitioner(nn.Module):
    """Transitioner for the 20-label gene-prediction HMM."""

    def __init__(
        self,
        k=1,
        num_models=1,
        initial_exon_len=100,
        initial_intron_len=10000,
        initial_ir_len=10000,
        initial_utr_len=60,
        initial_utr_intron_len=None,
        starting_distribution_init="zeros",
        starting_distribution_trainable=True,
        transitions_trainable=True,
        direct_transition_to_start_prob=0.1,
        stop_to_utr_prob=0.9,
        device=None,
        **kwargs,
    ):
        super(GenePredMultiHMMTransitioner, self).__init__(**kwargs)
        self.k = k
        self.num_models = num_models
        self.num_states = expanded_num_states(k)
        self.initial_exon_len = initial_exon_len
        self.initial_intron_len = initial_intron_len
        self.initial_ir_len = initial_ir_len
        self.initial_utr_len = initial_utr_len
        self.initial_utr_intron_len = (
            initial_utr_intron_len if initial_utr_intron_len is not None else initial_intron_len
        )
        self.starting_distribution_init = starting_distribution_init
        self.starting_distribution_trainable = starting_distribution_trainable
        self.transitions_trainable = transitions_trainable
        self.direct_transition_to_start_prob = direct_transition_to_start_prob
        self.stop_to_utr_prob = stop_to_utr_prob
        self.device = device
        self.reverse = False

        self.indices = self.make_transition_indices()
        self.num_transitions = len(self.indices)
        self.init = self.make_transition_init()
        self.transition_kernel = nn.Parameter(
            torch.tensor(self.init, dtype=torch.float32).unsqueeze(0),
            requires_grad=self.transitions_trainable,
        )

        start_kernel = torch.zeros(1, 1, self.num_states, dtype=torch.float32)
        if starting_distribution_init == "intergenic":
            start_kernel[..., 0] = 8.0
        elif starting_distribution_init != "zeros":
            start_kernel += 1.0
        self.starting_distribution_kernel = nn.Parameter(
            start_kernel,
            requires_grad=self.starting_distribution_trainable,
        )
        self.A = None
        self.A_transposed = None

    def _loop_probability(self, expected_length):
        expected_length = max(float(expected_length), 1.0)
        return 1.0 - 1.0 / expected_length

    def _duration_mass(self, expected_length):
        """Return the unnormalized stay weight for a geometric duration prior."""
        expected_length = max(float(expected_length), 1.0)
        return max(expected_length - 1.0, 1e-6)

    def _normalize(self, probabilities):
        total = float(sum(probabilities.values()))
        if total <= 0:
            raise ValueError("Transition probabilities must sum to a positive value.")
        return {state: prob / total for state, prob in probabilities.items()}

    def _base_transition_probabilities(self):
        state = BASE_GENE_PRED_STATE_TO_INDEX
        ir_loop = self._loop_probability(self.initial_ir_len)
        intron_loop = self._loop_probability(self.initial_intron_len)
        exon_continue = self._loop_probability(self.initial_exon_len)
        utr_stay = self._duration_mass(self.initial_utr_len)
        utr_intron_stay = self._duration_mass(self.initial_utr_intron_len)

        exon_exit = 1.0 - exon_continue

        return {
            state["IR"]: self._normalize(
                {
                    state["IR"]: ir_loop,
                    state["START"]: (1.0 - ir_loop) * self.direct_transition_to_start_prob,
                    state["5UTR"]: (1.0 - ir_loop) * (1.0 - self.direct_transition_to_start_prob),
                }
            ),
            state["intron0"]: {state["intron0"]: intron_loop, state["IE0"]: 1.0 - intron_loop},
            state["intron1"]: {state["intron1"]: intron_loop, state["IE1"]: 1.0 - intron_loop},
            state["intron2"]: {state["intron2"]: intron_loop, state["IE2"]: 1.0 - intron_loop},
            state["Exon0"]: {state["Exon1"]: exon_continue, state["EI0"]: exon_exit},
            state["Exon1"]: self._normalize(
                {
                    state["Exon2"]: exon_continue,
                    state["EI1"]: exon_exit * 0.5,
                    state["STOP"]: exon_exit * 0.5,
                }
            ),
            state["Exon2"]: {state["Exon0"]: exon_continue, state["EI2"]: exon_exit},
            state["START"]: {state["Exon1"]: 1.0},
            state["EI0"]: {state["intron0"]: 1.0},
            state["EI1"]: {state["intron1"]: 1.0},
            state["EI2"]: {state["intron2"]: 1.0},
            state["IE0"]: {state["Exon0"]: 1.0},
            state["IE1"]: {state["Exon1"]: 1.0},
            state["IE2"]: {state["Exon2"]: 1.0},
            state["STOP"]: {
                state["IR"]: 1.0 - self.stop_to_utr_prob,
                state["3UTR"]: self.stop_to_utr_prob,
            },
            # UTR entry remains optional, but when the model enters a UTR state
            # it now carries an explicit duration prior instead of collapsing to
            # a near-zero geometric mean.
            state["5UTR"]: self._normalize(
                {
                    state["5UTR"]: utr_stay,
                    state["START"]: 0.5,
                    state["UTR_EI"]: 0.5,
                }
            ),
            state["3UTR"]: self._normalize(
                {
                    state["3UTR"]: utr_stay,
                    state["IR"]: 0.5,
                    state["UTR_EI"]: 0.5,
                }
            ),
            state["UTR_EI"]: {state["UTR_INTRON"]: 1.0},
            state["UTR_INTRON"]: self._normalize(
                {
                    state["UTR_INTRON"]: utr_intron_stay,
                    state["UTR_IE"]: 1.0,
                }
            ),
            state["UTR_IE"]: {
                state["5UTR"]: 0.5,
                state["3UTR"]: 0.5,
            },
        }

    def make_transition_indices(self, model_index=0):
        indices = expanded_transition_edges(self.k)
        return np.concatenate(
            [np.full((len(indices), 1), model_index, dtype=np.int64), indices],
            axis=1,
        )

    def make_transition_init(self):
        base_probabilities = self._base_transition_probabilities()
        logits = []
        for _, src_state, dst_state in self.indices:
            if src_state == 0 and dst_state == 0:
                prob = base_probabilities[0][0]
            elif src_state == 0:
                base_dst = 1 + ((dst_state - 1) % (GENE_PRED_BASE_LABEL_DIM - 1))
                prob = base_probabilities[0][base_dst] / self.k
            else:
                base_src = 1 + ((src_state - 1) % (GENE_PRED_BASE_LABEL_DIM - 1))
                if dst_state == 0:
                    prob = base_probabilities[base_src][0]
                else:
                    base_dst = 1 + ((dst_state - 1) % (GENE_PRED_BASE_LABEL_DIM - 1))
                    prob = base_probabilities[base_src][base_dst]
            logits.append(np.log(max(prob, 1e-8)))
        return np.array(logits, dtype=np.float32)

    def recurrent_init(self):
        A = self.make_A()
        self.A = nn.Parameter(A, requires_grad=self.transitions_trainable)
        A_transposed = torch.transpose(self.A, 1, 2)
        self.A_transposed = nn.Parameter(A_transposed, requires_grad=self.transitions_trainable)
        show_value(A, "02.transitioner.A")
        show_value(A_transposed, "02.transitioner.A_transposed")

    def make_A_dense(self, values=None):
        if values is None:
            values = self.transition_kernel.view(-1)
        row_major_order = np.argsort([i * self.num_states + j for _, i, j in self.indices])
        ordered_indices = self.indices[row_major_order]
        ordered_values = values[row_major_order]
        dense_probs = make_transition_matrix_from_indices(
            ordered_indices[:, 1:],
            ordered_values,
            self.num_states,
        )
        dense_probs = dense_probs.unsqueeze(0)
        probs_vec = dense_probs[:, ordered_indices[:, 1], ordered_indices[:, 2]].squeeze(0)
        A_dense = torch.zeros(
            1,
            self.num_states,
            self.num_states,
            device=probs_vec.device,
            dtype=probs_vec.dtype,
        )
        A_dense[ordered_indices[:, 0], ordered_indices[:, 1], ordered_indices[:, 2]] = probs_vec
        return A_dense

    def make_A(self):
        return self.make_A_dense().repeat(self.num_models, 1, 1)

    def make_log_A(self):
        A_dense = self.make_A_dense()
        log_A = torch.where(
            A_dense > 0,
            torch.log(A_dense),
            torch.full_like(A_dense, -1e3),
        )
        return log_A.repeat(self.num_models, 1, 1)

    def make_initial_distribution(self):
        return F.softmax(self.starting_distribution_kernel, dim=-1).repeat(1, self.num_models, 1)

    def forward(self, inputs):
        if self.reverse:
            return torch.matmul(inputs, self.A_transposed)
        return torch.matmul(inputs, self.A)

    def get_prior_log_densities(self):
        return {"none": 0.0}

    def get_config(self):
        return {
            "k": self.k,
            "num_models": self.num_models,
            "initial_exon_len": self.initial_exon_len,
            "initial_intron_len": self.initial_intron_len,
            "initial_ir_len": self.initial_ir_len,
            "initial_utr_len": self.initial_utr_len,
            "initial_utr_intron_len": self.initial_utr_intron_len,
            "starting_distribution_init": self.starting_distribution_init,
            "starting_distribution_trainable": self.starting_distribution_trainable,
            "transitions_trainable": self.transitions_trainable,
            "direct_transition_to_start_prob": self.direct_transition_to_start_prob,
            "stop_to_utr_prob": self.stop_to_utr_prob,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class GenePredHMMTransitioner(GenePredMultiHMMTransitioner):
    def __init__(self, **kwargs):
        kwargs.setdefault("k", 1)
        super(GenePredHMMTransitioner, self).__init__(**kwargs)


class SimpleGenePredHMMTransitioner(GenePredMultiHMMTransitioner):
    def __init__(self, **kwargs):
        kwargs.setdefault("k", 1)
        super(SimpleGenePredHMMTransitioner, self).__init__(**kwargs)
