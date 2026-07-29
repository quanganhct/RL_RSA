import torch
from DRL.utils.logging import Logger
import datetime
import time

class RolloutWorker:
    """
    ============================================================
    PPO ROLLOUT WORKER FOR HIERARCHICAL RMSA
    ============================================================

    Collects:
        (obs,
         hierarchical action,
         logprobs,
         value,
         reward,
         done)

    using fixed-size PPO rollouts.

    IMPORTANT:
    ----------
    - NOT episode-based anymore
    - Supports truncated rollouts
    - Supports PPO mini-batching
    - One RMSA allocation = one PPO timestep

    ============================================================
    """

    def __init__(
        self,
        env,
        policy,
        logger,
        device="cpu"
    ):

        self.env = env
        self.policy = policy
        self.device = device
        self.logger:Logger = logger

        # --------------------------------------------------------
        # PERSISTENT ENV STATE
        # --------------------------------------------------------

        self.obs = self.env.customreset(False)

    # ============================================================
    # COLLECT PPO ROLLOUT
    # ============================================================

    def collect_rollout(self, buffer, deterministic=False):

        """
        Fills PPO rollout buffer until full.
        """

        rollout_info = {

            "service_blocking_rate": [],
            "episode_service_blocking_rate": [],
            "our_service_blocking_rate": [],
            "bit_rate_blocking_rate": [],
            "episode_bit_rate_blocking_rate": [],
            "avg_link_utilization": [],
            'num_accepted_request': [],
            'num_total_request': [],
            'done': []
        }

        while not buffer.is_full():

            obs =  self._to_device(self.obs)

            # =====================================================
            # STAGE 1: PATH ACTION
            # =====================================================

            with torch.no_grad():

                path_action, path_logprob, cache = (
                    self.policy.act_path(obs, deterministic)
                )

            obs_after_path, _ = self.env.step_path(
                obs,
                path_action
            )

            # =====================================================
            # STAGE 2: MODULATION ACTION
            # =====================================================

            with torch.no_grad():

                mod_action, mod_logprob, mod_emb = (
                    self.policy.act_modulation(
                        self._to_device(obs_after_path),
                        cache['selected_path_emb'],
                        deterministic
                    )
                )

            cache["selected_mod_emb"] = mod_emb

            obs_after_mod, _ = self.env.step_modulation(
                obs_after_path,
                mod_action
            )

            # =====================================================
            # STAGE 3: SLOT ACTION
            # =====================================================

            with torch.no_grad():

                slot_action, slot_logprob = (
                    self.policy.act_slot(
                        self._to_device(obs_after_mod),
                        cache,
                        deterministic
                    )
                )

            # =====================================================
            # CRITIC VALUE
            # =====================================================

            with torch.no_grad():

                value = self.policy.evaluate_value(self._to_device(obs))

            # =====================================================
            # ENV STEP
            # =====================================================
            start = time.time()
            next_obs, reward, done, info = self.env.step(
                slot_action
            )
            self.logger.log_str("ENV step + next obs: %s seconds"%((time.time()-start)))
            # =====================================================
            # STORE PPO TRANSITION
            # =====================================================

            buffer.add_transition(

                obs=obs,

                path_action=path_action,
                mod_action=mod_action,
                slot_action=slot_action,

                path_logprob=path_logprob,
                mod_logprob=mod_logprob,
                slot_logprob=slot_logprob,

                value=value,

                reward=reward,
                done=done
            )

            # =====================================================
            # LOGGING
            # =====================================================
            for key in rollout_info.keys():
                
                rollout_info[key].append(
                    info[key]
                )

            # rollout_info["bit_rate_blocking_rate"].append(
            #     info["bit_rate_blocking_rate"]
            # )
            # rollout_info["bit_rate_blocking_rate"].append(
            #     info["bit_rate_blocking_rate"]
            # )

            # rollout_info["bit_rate_blocking_rate"].append(
            #     info["bit_rate_blocking_rate"]
            # )


            # rollout_info["avg_link_utilization"].append(
            #     info["avg_link_utilization"]
            # )

            # =====================================================
            # NEXT STATE
            # =====================================================

            if done:
                # buffer.last_obs = obs

                self.obs = self.env.customreset(False)
                print(f'done with buffer size = {buffer.ptr}')

            else:

                self.obs = next_obs
        print(f"full when buffer is size is {buffer.ptr}")
        # =========================================================
        # BOOTSTRAP VALUE
        # =========================================================

        with torch.no_grad():

            last_value = self.policy.evaluate_value(
                self._to_device(next_obs) #self.obs
                
            )

        return last_value, rollout_info
    
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