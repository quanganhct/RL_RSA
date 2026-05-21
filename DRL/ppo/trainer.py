import torch
import torch.optim as optim

from DRL.ppo.losses import compute_hierarchical_loss
from DRL.ppo.gae import compute_hierarchical_gae, compute_gae


class PPOTrainer:

    def __init__(
        self,
        policy,
        logger=None,
        lr=3e-4,
        clip_eps=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        ppo_epochs=4,
        mini_batch_size=256,
        device="cpu"
    ):

        self.policy = policy
        self.device = device
        self.logger = logger

        self.optimizer = optim.Adam(policy.parameters(), lr=lr)

        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size

        self.global_step = 0

    # ============================================================
    # TRAIN STEP (ONE ROLLOUT UPDATE)
    # ============================================================

    def train_step(self, batch, last_value):

        self.global_step += 1

        # --------------------------------------------------------
        # MOVE TO DEVICE
        # --------------------------------------------------------

        batch = self._to_device(batch)

        # --------------------------------------------------------
        # OLD VALUES (fixed policy snapshot)
        # --------------------------------------------------------

        with torch.no_grad():
            outputs = self.policy.forward_ppo(batch)
            values = outputs["value"].squeeze(-1)

        # --------------------------------------------------------
        # GAE (NOW CLEAN SEPARATE MODULE)
        # --------------------------------------------------------

        advantages, returns = compute_gae(
            rewards=batch["rewards"],
            values=values,
            dones=batch["dones"],
            last_value=last_value
        )
        
        

        batch["advantages"] = advantages
        batch["returns"] = returns

        # --------------------------------------------------------
        # MINIBATCH PPO UPDATE
        # --------------------------------------------------------

        stats_accum = {}

        for epoch in range(self.ppo_epochs):

            indices = torch.randperm(len(values))

            for start in range(0, len(values), self.mini_batch_size):

                end = start + self.mini_batch_size
                mb_idx = indices[start:end]

                mb_batch = self._index_batch(batch, mb_idx)

                outputs = self.policy.forward_ppo(mb_batch)

                loss, stats = compute_hierarchical_loss(
                    outputs=outputs,
                    batch=mb_batch,
                    advantages=advantages[mb_idx],
                    returns=returns[mb_idx],
                    config={
                        "clip_eps": self.clip_eps,
                        "value_coef": self.value_coef,
                        "entropy_coef": self.entropy_coef
                    }
                )

                self.optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(),
                    0.5
                )

                self.optimizer.step()

                self._accumulate_stats(stats_accum, stats)

        # --------------------------------------------------------
        # LOGGING
        # --------------------------------------------------------

        if self.logger is not None:
            self.logger.log(self.global_step, stats_accum)

        return stats_accum

    # ============================================================
    # DEVICE HANDLING
    # ============================================================


    def _to_device(self, batch):
        for k in batch:
            if torch.is_tensor(batch[k]):
                batch[k] = batch[k].to(self.device)
                
    # def _to_device(self, batch):

    #     def move(x):

    #         if torch.is_tensor(x):
    #             return x.to(self.device)

    #         if isinstance(x, dict):
    #             return {k: move(v) for k, v in x.items()}

    #         if isinstance(x, list):
    #             return [move(v) for v in x]

    #         return x

    #     return move(batch)

    # ============================================================
    # MINIBATCH INDEXING
    # ============================================================

    def _index_batch(self, batch, idx):

        new_batch = {}

        for k, v in batch.items():

            if torch.is_tensor(v):

                new_batch[k] = v[idx]

            else:

                new_batch[k] = v

        return new_batch

    # ============================================================
    # STATS ACCUMULATION
    # ============================================================

    def _accumulate_stats(self, acc, stats):

        for k, v in stats.items():

            if k not in acc:
                acc[k] = []

            acc[k].append(v)

    # ============================================================
    # DEVICE HANDLING
    # ============================================================

    def _to_device(self, batch):

        def recursive_move(obj):

            if torch.is_tensor(obj):
                return obj.to(self.device)

            if isinstance(obj, dict):
                return {k: recursive_move(v) for k, v in obj.items()}

            if isinstance(obj, list):
                return [recursive_move(v) for v in obj]

            return obj

        return recursive_move(batch)