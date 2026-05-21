import torch
import numpy as np


class HierarchicalRolloutBuffer:

    def __init__(
        self,
        rollout_size,
        mini_batch_size,
        device="cpu"
    ):

        self.rollout_size = rollout_size
        self.mini_batch_size = mini_batch_size
        self.device = device

        self.reset()

    # ============================================================
    # RESET
    # ============================================================

    def reset(self):

        # --------------------------------------------------------
        # OBSERVATIONS
        # --------------------------------------------------------

        self.edge_features = []
        self.edge_index = []
        self.path_features = []
        self.mod_features = []

        self.path_masks = []
        self.mod_masks = []
        self.slot_masks = []

        self.candidate_paths = []

        # --------------------------------------------------------
        # ACTIONS
        # --------------------------------------------------------

        self.path_actions = []
        self.mod_actions = []
        self.slot_actions = []

        # --------------------------------------------------------
        # LOGPROBS
        # --------------------------------------------------------

        self.path_logprobs = []
        self.mod_logprobs = []
        self.slot_logprobs = []

        # --------------------------------------------------------
        # VALUES
        # --------------------------------------------------------

        self.values = []

        # --------------------------------------------------------
        # REWARDS
        # --------------------------------------------------------

        self.rewards = []
        self.dones = []

        self.ptr = 0
        
        self.last_obs = None

    # ============================================================
    # ADD TRANSITION
    # ============================================================

    def add_transition(
        self,
        obs,
        path_action,
        mod_action,
        slot_action,
        path_logprob,
        mod_logprob,
        slot_logprob,
        value,
        reward,
        done
    ):

        # --------------------------------------------------------
        # OBS
        # --------------------------------------------------------

        self.edge_features.append(
            obs["edge_features"].detach().cpu()
        )
        self.edge_index.append(
            obs["edge_index"].detach().cpu()
        )

        self.path_features.append(
            obs["path_features"].detach().cpu()
        )

        self.mod_features.append(
            obs["mod_features"].detach().cpu()
        )

        self.path_masks.append(
            obs["action_masks"]["path"].detach().cpu()
        )

        self.mod_masks.append(
            obs["action_masks"]["mod"].detach().cpu()
        )

        self.slot_masks.append(
            obs["action_masks"]["slot"].detach().cpu()
        )

        self.candidate_paths.append(
            obs["candidate_paths"].detach().cpu()
        )

        # --------------------------------------------------------
        # ACTIONS
        # --------------------------------------------------------

        self.path_actions.append(path_action.detach().cpu())
        self.mod_actions.append(mod_action.detach().cpu())
        self.slot_actions.append(slot_action.detach().cpu())

        # --------------------------------------------------------
        # LOGPROBS
        # --------------------------------------------------------

        self.path_logprobs.append(path_logprob.detach().cpu())
        self.mod_logprobs.append(mod_logprob.detach().cpu())
        self.slot_logprobs.append(slot_logprob.detach().cpu())

        # --------------------------------------------------------
        # VALUE
        # --------------------------------------------------------

        self.values.append(value.detach().view(1).cpu())

        # --------------------------------------------------------
        # REWARD / DONE
        # --------------------------------------------------------

        self.rewards.append(float(reward))
        self.dones.append(float(done))

        self.ptr += 1

    # ============================================================
    # STATUS
    # ============================================================

    def is_full(self):

        return self.ptr >= self.rollout_size

    def __len__(self):

        return self.ptr

    def clear(self):

        self.reset()

    # ============================================================
    # BUILD FULL PPO BATCH
    # ============================================================

    def get_batch(self):

        batch = {

            # ----------------------------------------------------
            # OBS
            # ----------------------------------------------------
           
               "edge_features":
                torch.cat(self.edge_features, dim=0).to(self.device),
                "edge_index":
                 torch.cat(self.edge_index, dim=0).to(self.device),

            "path_features":
                torch.cat(self.path_features, dim=0).to(self.device),

            "mod_features":
                torch.cat(self.mod_features, dim=0).to(self.device),

            "candidate_paths":
                torch.cat(self.candidate_paths, dim=0).to(self.device),

            # ----------------------------------------------------
            # MASKS
            # ----------------------------------------------------

            "action_masks": {

                "path":
                    torch.cat(self.path_masks, dim=0).to(self.device),

                "mod":
                    torch.cat(self.mod_masks, dim=0).to(self.device),

                "slot":
                    torch.cat(self.slot_masks, dim=0).to(self.device)
            },

            # ----------------------------------------------------
            # ACTIONS
            # ----------------------------------------------------

            "path_actions":
                torch.cat(self.path_actions, dim=0).long().to(self.device),

            "mod_actions":
                torch.cat(self.mod_actions, dim=0).long().to(self.device),

            "slot_actions":
                torch.cat(self.slot_actions, dim=0).long().to(self.device),

            # ----------------------------------------------------
            # OLD LOGPROBS
            # ----------------------------------------------------

            "path_logprobs":
                torch.cat(self.path_logprobs, dim=0).to(self.device),

            "mod_logprobs":
                torch.cat(self.mod_logprobs, dim=0).to(self.device),

            "slot_logprobs":
                torch.cat(self.slot_logprobs, dim=0).to(self.device),

            # ----------------------------------------------------
            # VALUES
            # ----------------------------------------------------

            "values":
                torch.cat(self.values).squeeze(-1).to(self.device),

            # ----------------------------------------------------
            # REWARDS
            # ----------------------------------------------------

            "rewards":
                torch.tensor(
                    self.rewards,
                    dtype=torch.float32
                ).to(self.device),

            "dones":
                torch.tensor(
                    self.dones,
                    dtype=torch.float32
                ).to(self.device),
                
            "last_obs": self.last_obs
        }

        return batch

    # ============================================================
    # MINIBATCH ITERATOR
    # ============================================================

    def iterate_minibatches(
        self,
        advantages,
        returns
    ):

        indices = np.arange(self.ptr)

        np.random.shuffle(indices)

        for start in range(
            0,
            self.ptr,
            self.mini_batch_size
        ):

            end = start + self.mini_batch_size

            mb_idx = indices[start:end]

            yield self._build_minibatch(
                mb_idx,
                advantages,
                returns
            )

    # ============================================================
    # BUILD MINIBATCH
    # ============================================================

    def _build_minibatch(
        self,
        idx,
        advantages,
        returns
    ):

        full_batch = self.get_batch()

        mini_batch = {}

        for k, v in full_batch.items():

            if isinstance(v, dict):

                mini_batch[k] = {
                    kk: vv[idx]
                    for kk, vv in v.items()
                }

            else:
                mini_batch[k] = v[idx]

        mini_batch["advantages"] = advantages[idx]
        mini_batch["returns"] = returns[idx]

        return mini_batch