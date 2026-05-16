# ============================================================
# FULL HIERARCHICAL INFERENCE
# ============================================================
import torch
@torch.no_grad()
def act(
    self,
    obs,
    deterministic=False
):

    """
    ========================================================
    FULL HIERARCHICAL RMSA DECISION PIPELINE
    ========================================================

    Stages:

        1. path selection
        2. modulation selection
        3. slot selection

    ========================================================
    INPUT
    ========================================================

    obs dict should contain:

        edge_features
        edge_index
        candidate_paths
        path_features
        slot_features

        action_masks:
            path
            mod
            slot

    ========================================================
    OUTPUT
    ========================================================

    {
        "path": int or tensor,
        "modulation": int or tensor,
        "slot": int or tensor,

        "logprob_path": tensor,
        "logprob_mod": tensor,
        "logprob_slot": tensor,

        "value": tensor
    }

    ========================================================
    """

    device = next(self.parameters()).device

    obs = self._move_obs_to_device(
        obs,
        device
    )

    # =======================================================
    # SINGLE/BATCH DETECTION
    # =======================================================

    single_sample = False

    if obs["candidate_paths"].dim() == 2:
        """
        single sample mode:
            candidate_paths = [K,L]

        Convert to:
            [1,K,L]
        """

        single_sample = True

        obs = self._add_batch_dim(obs)

    B = obs["candidate_paths"].shape[0]

    # =======================================================
    # EDGE ENCODER
    # =======================================================

    edge_emb = self.encoder(

        obs["edge_features"],
        obs["edge_index"]
    )

    # =======================================================
    # PATH ENCODER
    # =======================================================

    path_emb, _ = self.path_encoder(

        edge_emb,
        obs["candidate_paths"]
    )

    # path_emb:
    # [B,K,D]

    # =======================================================
    # PATH POLICY
    # =======================================================

    path_logits = self.path_policy(

        path_embeddings=path_emb,

        path_mask=
            obs["action_masks"]["path"],

        path_features=
            obs.get("path_features", None)
    )

    path_dist = torch.distributions.Categorical(
        logits=path_logits
    )

    if deterministic:

        path_action = torch.argmax(
            path_logits,
            dim=-1
        )

    else:

        path_action = path_dist.sample()

    path_logprob = path_dist.log_prob(
        path_action
    )

    # =======================================================
    # SELECT PATH CONTEXT
    # =======================================================

    batch_idx = torch.arange(
        B,
        device=device
    )

    selected_path_emb = path_emb[
        batch_idx,
        path_action
    ]

    # -------------------------------------------------------
    # SELECT PATH FEATURES
    # -------------------------------------------------------

    path_features = obs.get(
        "path_features",
        None
    )

    if path_features is not None:

        # expected:
        # [B,K,F]

        selected_path_features = path_features[
            batch_idx,
            path_action
        ]

    else:

        selected_path_features = None

    # =======================================================
    # MODULATION POLICY
    # =======================================================

    mod_logits, mod_context = self.mod_policy(

        selected_path_emb,
        selected_path_features,

        obs["action_masks"]["mod"]
    )

    mod_dist = torch.distributions.Categorical(
        logits=mod_logits
    )

    if deterministic:

        mod_action = torch.argmax(
            mod_logits,
            dim=-1
        )

    else:

        mod_action = mod_dist.sample()

    mod_logprob = mod_dist.log_prob(
        mod_action
    )

    # =======================================================
    # MODULATION EMBEDDING
    # =======================================================

    mod_emb = self.mod_policy.get_modulation_embedding(
        mod_action
    )

    # =======================================================
    # SLOT POLICY
    # =======================================================

    slot_logits, slot_emb = self.slot_policy(

        obs["slot_features"],

        selected_path_emb,

        mod_emb,

        obs["action_masks"]["slot"]
    )

    slot_dist = torch.distributions.Categorical(
        logits=slot_logits
    )

    if deterministic:

        slot_action = torch.argmax(
            slot_logits,
            dim=-1
        )

    else:

        slot_action = slot_dist.sample()

    slot_logprob = slot_dist.log_prob(
        slot_action
    )

    # =======================================================
    # VALUE ESTIMATION
    # =======================================================

    value = self.critic(

        edge_embeddings=edge_emb,

        path_embedding=
            selected_path_emb,

        mod_embedding=
            mod_emb,

        slot_embedding=
            slot_emb.mean(dim=1)
            if slot_emb.dim() == 3
            else slot_emb
    )

    # =======================================================
    # REMOVE BATCH DIM (SINGLE SAMPLE)
    # =======================================================

    if single_sample:

        return {

            "path":
                path_action.item(),

            "modulation":
                mod_action.item(),

            "slot":
                slot_action.item(),

            "logprob_path":
                path_logprob.item(),

            "logprob_mod":
                mod_logprob.item(),

            "logprob_slot":
                slot_logprob.item(),

            "value":
                value.item()
        }

    # =======================================================
    # BATCH RETURN
    # =======================================================

    return {

        "path":
            path_action,

        "modulation":
            mod_action,

        "slot":
            slot_action,

        "logprob_path":
            path_logprob,

        "logprob_mod":
            mod_logprob,

        "logprob_slot":
            slot_logprob,

        "value":
            value
    }


# ============================================================
# HELPERS
# ============================================================

def _move_obs_to_device(
    self,
    obj,
    device
):

    if torch.is_tensor(obj):

        return obj.to(device)

    elif isinstance(obj, dict):

        return {
            k: self._move_obs_to_device(v, device)
            for k, v in obj.items()
        }

    elif isinstance(obj, list):

        return [
            self._move_obs_to_device(v, device)
            for v in obj
        ]

    else:

        return obj


def _add_batch_dim(
    self,
    obs
):

    out = {}

    for k, v in obs.items():

        if isinstance(v, dict):

            out[k] = self._add_batch_dim(v)

        elif torch.is_tensor(v):

            out[k] = v.unsqueeze(0)

        else:

            out[k] = v

    return out