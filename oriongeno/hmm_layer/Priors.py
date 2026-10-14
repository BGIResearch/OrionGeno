import warnings

import torch
import torch.nn as nn


LEGACY_PROFILE_HMM_MESSAGE = (
    "The legacy Profile-HMM prior API is deprecated and will be removed in "
    "OrionGeno 0.2.0. Its trained mixture-model assets and loader were never "
    "distributed with this repository. Use GenePredHMMLayer for gene prediction, "
    "or pass a fully implemented custom prior to ProfileHMMTransitioner."
)


class ProfileHMMTransitionPrior(nn.Module):
    """Deprecated transition prior for the incomplete legacy Profile-HMM API."""

    def __init__(
        self,
        match_comp=1,
        insert_comp=1,
        delete_comp=1,
        alpha_flank=7000,
        alpha_single=1e9,
        alpha_global=1e4,
        alpha_flank_compl=1,
        alpha_single_compl=1,
        alpha_global_compl=1,
        epsilon=1e-16,
        **kwargs,
    ):
        super(ProfileHMMTransitionPrior, self).__init__(**kwargs)
        warnings.warn(LEGACY_PROFILE_HMM_MESSAGE, DeprecationWarning, stacklevel=2)
        self.match_comp = match_comp
        self.insert_comp = insert_comp
        self.delete_comp = delete_comp

        self.alpha_flank = alpha_flank
        self.alpha_single = alpha_single
        self.alpha_global = alpha_global
        self.alpha_flank_compl = alpha_flank_compl
        self.alpha_single_compl = alpha_single_compl
        self.alpha_global_compl = alpha_global_compl
        self.epsilon = epsilon

    def build(self, input_shape=None):
        """Fail explicitly because the historical prior assets are unavailable."""
        del input_shape
        raise RuntimeError(LEGACY_PROFILE_HMM_MESSAGE)

    def forward(self, probs_list, flank_init_prob):
        """Evaluate transition-prior log densities for each model."""
        match_dirichlet = []
        insert_dirichlet = []
        delete_dirichlet = []
        flank_prior = []
        hit_prior = []
        global_prior = []
        for i, probs in enumerate(probs_list):
            log_probs = {part_name: torch.log(p) for part_name, p in probs.items()}

            p_match = (
                torch.stack(
                    [
                        probs["match_to_match"],
                        probs["match_to_insert"],
                        probs["match_to_delete"][1:],
                    ],
                    dim=-1,
                )
                + self.epsilon
            )
            p_match_sum = torch.sum(p_match, dim=-1, keepdim=True)
            match_dirichlet.append(
                torch.sum(self.match_dirichlet.log_prob(p_match / p_match_sum))
            )

            p_insert = torch.stack(
                [probs["insert_to_match"], probs["insert_to_insert"]], dim=-1
            )
            insert_dirichlet.append(torch.sum(self.insert_dirichlet.log_prob(p_insert)))

            p_delete = torch.stack(
                [probs["delete_to_match"][:-1], probs["delete_to_delete"]], dim=-1
            )
            delete_dirichlet.append(torch.sum(self.delete_dirichlet.log_prob(p_delete)))

            flank = (self.alpha_flank - 1) * log_probs["unannotated_segment_loop"]
            flank += (self.alpha_flank - 1) * log_probs["right_flank_loop"]
            flank += (self.alpha_flank - 1) * log_probs["left_flank_loop"]
            flank += (self.alpha_flank - 1) * log_probs["end_to_right_flank"]
            flank += (self.alpha_flank - 1) * torch.log(flank_init_prob[i])
            flank += (self.alpha_flank_compl - 1) * log_probs[
                "unannotated_segment_exit"
            ]
            flank += (self.alpha_flank_compl - 1) * log_probs["right_flank_exit"]
            flank += (self.alpha_flank_compl - 1) * log_probs["left_flank_exit"]
            flank += (self.alpha_flank_compl - 1) * torch.log(
                probs["end_to_unannotated_segment"] + probs["end_to_terminal"]
            )
            flank += (self.alpha_flank_compl - 1) * torch.log(1 - flank_init_prob[i])
            flank_prior.append(torch.squeeze(flank))

            hit = (self.alpha_single - 1) * torch.log(
                probs["end_to_right_flank"] + probs["end_to_terminal"]
            )
            hit += (self.alpha_single_compl - 1) * torch.log(
                probs["end_to_unannotated_segment"]
            )
            hit_prior.append(torch.squeeze(hit))

            div = 1 / torch.maximum(
                torch.tensor(self.epsilon), 1 - probs["match_to_delete"][0]
            )
            btm = probs["begin_to_match"] / div
            enex = torch.unsqueeze(btm, 1) * torch.unsqueeze(probs["match_to_end"], 0)
            enex = torch.tril(enex)
            log_enex = torch.log(torch.maximum(torch.tensor(self.epsilon), 1 - enex))
            log_enex_compl = torch.log(torch.maximum(torch.tensor(self.epsilon), enex))
            glob = (self.alpha_global - 1) * (torch.sum(log_enex) - log_enex[0, -1])
            glob += (self.alpha_global_compl - 1) * (
                torch.sum(log_enex_compl) - log_enex_compl[0, -1]
            )
            global_prior.append(glob)
        prior_val = {
            "match_prior": match_dirichlet,
            "insert_prior": insert_dirichlet,
            "delete_prior": delete_dirichlet,
            "flank_prior": flank_prior,
            "hit_prior": hit_prior,
            "global_prior": global_prior,
        }
        prior_val = {k: torch.stack(v) for k, v in prior_val.items()}
        return prior_val

    def get_config(self):
        config = {
            "match_comp": self.match_comp,
            "insert_comp": self.insert_comp,
            "delete_comp": self.delete_comp,
            "alpha_flank": self.alpha_flank,
            "alpha_single": self.alpha_single,
            "alpha_global": self.alpha_global,
            "alpha_flank_compl": self.alpha_flank_compl,
            "alpha_single_compl": self.alpha_single_compl,
            "alpha_global_compl": self.alpha_global_compl,
            "epsilon": self.epsilon,
        }
        return config

    def __repr__(self):
        return f"ProfileHMMTransitionPrior(match_comp={self.match_comp}, insert_comp={self.insert_comp}, delete_comp={self.delete_comp}, alpha_flank={self.alpha_flank}, alpha_single={self.alpha_single}, alpha_global={self.alpha_global}, alpha_flank_compl={self.alpha_flank_compl}, alpha_single_compl={self.alpha_single_compl}, alpha_global_compl={self.alpha_global_compl})"
