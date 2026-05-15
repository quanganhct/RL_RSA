import torch
import torch.optim as optim

from ppo.losses_good import compute_hierarchical_loss
from ppo.gae_good import compute_hierarchical_gae


class PPOTrainer:
    """
    ============================================================
    HIERARCHICAL PPO TRAINER FOR RMSA
    ============================================================

    Pipeline:

        rollout_worker → buffer → GAE → forward_ppo → loss → update

    ============================================================
    """

    def __init__(
        self,
        policy,
        lr=3e-4,
        clip_eps=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        device="cpu"
    ):

        self.policy = policy
        self.device = device

        self.optimizer = optim.Adam(policy.parameters(), lr=lr)

        self.config = {
            "clip_eps": clip_eps,
            "value_coef": value_coef,
            "entropy_coef": entropy_coef
        }

    # ============================================================
    # TRAIN STEP
    # ============================================================

    def train_step(self, batch):

        # --------------------------------------------------------
        # MOVE TO DEVICE
        # --------------------------------------------------------
        print(f"(batch)------------------------------ = {len(batch)}")
        batch = self._to_device(batch)
        print(f"(batch) +++++++++++++++++++++++++++++= {len(batch)}")
        # --------------------------------------------------------
        # VALUE ESTIMATION (FROM CURRENT POLICY)
        # --------------------------------------------------------

        with torch.no_grad():

            # NOTE:
            # forward_ppo returns logits + value
            # print(f"batch = {batch}")
            outputs = self.policy.forward_ppo(batch)

            values = outputs["value"].squeeze()

        # --------------------------------------------------------
        # GAE COMPUTATION (HIERARCHICAL)
        # --------------------------------------------------------

        advantages, returns = compute_hierarchical_gae(
            rewards=batch["rewards"],
            values=values,
            dones=batch["dones"],
            stage_ids=batch["stage_ids"]
        )

        # --------------------------------------------------------
        # NORMALIZE ADVANTAGES
        # --------------------------------------------------------

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # --------------------------------------------------------
        # FORWARD PASS (WITH GRADIENT)
        # --------------------------------------------------------

        outputs = self.policy.forward_ppo(batch)

        # --------------------------------------------------------
        # LOSS COMPUTATION
        # --------------------------------------------------------

        loss, stats = compute_hierarchical_loss(
            outputs=outputs,
            batch=batch,
            advantages={
                "path": advantages,
                "mod": advantages,
                "slot": advantages
            },
            returns=returns,
            config=self.config
        )

        # --------------------------------------------------------
        # BACKPROP
        # --------------------------------------------------------

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.policy.parameters(),
            max_norm=0.5
        )

        self.optimizer.step()

        return stats

    # ============================================================
    # DEVICE HANDLING
    # ============================================================
    def _to_device(self, batch):
    
        def move(x):
            if torch.is_tensor(x):
                return x.to(self.device)
            return x
    
        def recursive_move(obj):
            if torch.is_tensor(obj):
                return obj.to(self.device)
    
            if isinstance(obj, dict):
                return {k: recursive_move(v) for k, v in obj.items()}
    
            if isinstance(obj, list):
                return [recursive_move(v) for v in obj]
    
            return obj
    
        return recursive_move(batch)
    # def _to_device(self, batch):

    #     def move(x):
    #         if torch.is_tensor(x):
    #             return x.to(self.device)
    #         return x

    #     return {k: move(v) for k, v in batch.items()}