import torch

from tqdm import tqdm

def rl_train(env, agent, alg='PPO', max_steps: int | None=10000):

    pbar = tqdm(total=max_steps) if max_steps is not None else None
    step_count = 0

    obs, info = env.reset()
    done_prev = 0

    while True:

        with torch.no_grad():
            action, value, log_prob = agent.forward(obs)

        next_obs, reward, done, trunc, info = env.step(action)

        if alg == 'PPO':
            b_full = agent.add_rollout_step(obs, done_prev, action, value, log_prob, reward)
            if b_full:
                with torch.no_grad():
                    _, next_val, _ = agent.forward(next_obs)
                agent.add_lastv_done(next_val, done)
                agent.train()

        done_prev = done    

        if max_steps is not None and step_count > max_steps:
            break

        step_count += 1
        if pbar is not None: pbar.update(1)

        


