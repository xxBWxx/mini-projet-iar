import os
import numpy as np

import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import DDPG, TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnRewardThreshold,
)
from stable_baselines3.common.vec_env import DummyVecEnv

import torch
from typing import Tuple

SEED = 42
TIMESTEPS = 60_000  # adjustable
EVAL_EPISODES = 20
GAMMA = 0.99
ENV_ID = "LunarLanderContinuous-v3"

__all__ = ["SEED", "TIMESTEPS", "EVAL_EPISODES", "GAMMA", "ENV_ID"]


def make_env(seed=SEED):
    def _thunk():
        env = gym.make(ENV_ID)
        env = TimeLimit(env, max_episode_steps=1000)
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env = Monitor(env)
        return env

    return _thunk


def train_model(alg: str, logdir: str, timesteps: int) -> Tuple[object, str]:
    env = DummyVecEnv([make_env(SEED)])
    n_actions = env.action_space.shape[-1]

    # exploration noise for deterministic policies
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
    )

    if alg == "DDPG":
        model = DDPG(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            buffer_size=1_000_000,
            batch_size=256,
            gamma=GAMMA,
            tau=0.005,
            train_freq=(1, "step"),
            gradient_steps=1,
            action_noise=action_noise,
            tensorboard_log=logdir,
            seed=SEED,
            verbose=1,
        )
    elif alg == "TD3":
        model = TD3(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            buffer_size=1_000_000,
            batch_size=256,
            gamma=GAMMA,
            tau=0.005,
            train_freq=(1, "step"),
            gradient_steps=1,
            action_noise=action_noise,
            tensorboard_log=logdir,
            seed=SEED,
            verbose=1,
            policy_delay=2,
            target_policy_noise=0.2,
            target_noise_clip=0.5,
        )
    else:
        raise ValueError("alg must be 'DDPG' or 'TD3'")

    # evaluate while training
    eval_env = make_env(SEED + 1)()
    stop_cb = StopTrainingOnRewardThreshold(reward_threshold=200, verbose=1)
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(logdir, f"{alg}_best"),
        log_path=os.path.join(logdir, f"{alg}_eval"),
        eval_freq=10_000,
        n_eval_episodes=5,
        deterministic=True,
        callback_on_new_best=stop_cb,
    )

    model.learn(total_timesteps=timesteps, callback=eval_cb)

    model.save(os.path.join(logdir, f"{alg}_final"))
    env.close()
    eval_env.close()
    return model, os.path.join(logdir, f"{alg}_final.zip")


@torch.no_grad()
def critic_q_estimate(model, obs: np.ndarray, act: np.ndarray):
    """
    Returns:
      - For DDPG: single Q estimate (N,)
      - For TD3: (q1, q2, q_min) each (N,)
    Works with SB3 >= 1.8 style modules.
    """
    device = model.device
    obs_t = torch.as_tensor(obs).float().to(device)
    act_t = torch.as_tensor(act).float().to(device)

    # DDPG: model.critic
    # TD3: model.critic, model.critic_target with two Q nets internally; expose via forward with return both
    try:
        # TD3 exposes two critics inside model.critic
        out = model.critic(obs_t, act_t)
        if isinstance(out, tuple) or (hasattr(out, "__len__") and len(out) == 2):
            q1, q2 = out
            q1 = q1.flatten().cpu().numpy()
            q2 = q2.flatten().cpu().numpy()
            return q1, q2, np.minimum(q1, q2)
        else:
            q = out.flatten().cpu().numpy()
            return q
    except Exception:
        if hasattr(model.policy, "q_net"):
            q = model.policy.q_net(obs_t, act_t).flatten().cpu().numpy()
            return q
        q1 = model.policy.qf1(obs_t, act_t).flatten().cpu().numpy()
        q2 = model.policy.qf2(obs_t, act_t).flatten().cpu().numpy()
        return q1, q2, np.minimum(q1, q2)


def rollout_and_bias(model, episodes=10, gamma=GAMMA, seed=SEED + 123):
    """Run deterministic rollouts, collect (obs, action, return-to-go) and compare with critic estimates."""
    env = gym.make(ENV_ID)
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    returns = []
    all_obs, all_act, all_rtgs = [], [], []
    for _ in range(episodes):
        obs, info = env.reset(seed=rng.integers(0, 10_000))
        done = False
        traj_obs, traj_act, traj_rew = [], [], []
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            next_obs, reward, terminated, truncated, info = env.step(action)
            traj_obs.append(obs)
            traj_act.append(action)
            traj_rew.append(reward)
            obs = next_obs
            done = terminated or truncated

        # compute per-step return-to-go (MC target) for comparison
        G = 0.0
        rtgs = []
        for r in reversed(traj_rew):
            G = r + gamma * G
            rtgs.append(G)
        rtgs = list(reversed(rtgs))
        returns.append(sum(traj_rew))
        all_obs.extend(traj_obs)
        all_act.extend(traj_act)
        all_rtgs.extend(rtgs)
    env.close()
    all_obs = np.array(all_obs, dtype=np.float32)
    all_act = np.array(all_act, dtype=np.float32)
    all_rtgs = np.array(all_rtgs, dtype=np.float32)

    # critic estimates at (s,a)
    q_est = critic_q_estimate(model, all_obs, all_act)
    if isinstance(q_est, tuple):
        q1, q2, qmin = q_est

        # compare to MC return-to-go
        bias_q1 = np.mean(q1 - all_rtgs)
        bias_q2 = np.mean(q2 - all_rtgs)
        bias_qmin = np.mean(qmin - all_rtgs)
        return {
            "episode_returns": returns,
            "mean_return": float(np.mean(returns)),
            "std_return": float(np.std(returns)),
            "bias_q1": float(bias_q1),
            "bias_q2": float(bias_q2),
            "bias_qmin": float(bias_qmin),
        }
    else:
        q = q_est
        bias = np.mean(q - all_rtgs)
        return {
            "episode_returns": returns,
            "mean_return": float(np.mean(returns)),
            "std_return": float(np.std(returns)),
            "bias": float(bias),
        }
