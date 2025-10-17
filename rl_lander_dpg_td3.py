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
        env.reset(seed=int(seed))
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

    eval_env = make_env(SEED + 1)()

    # pick an eval freq that always triggers (at least once) before training ends
    effective_eval_freq = max(1, min(10_000, max(1, timesteps // 5)))

    stop_cb = StopTrainingOnRewardThreshold(reward_threshold=200, verbose=1)

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(logdir, f"{alg}_best"),
        log_path=os.path.join(logdir, f"{alg}_eval"),
        eval_freq=effective_eval_freq,
        n_eval_episodes=5,
        deterministic=True,
        callback_on_new_best=stop_cb,
    )

    model.learn(total_timesteps=timesteps, callback=eval_cb)

    save_path = os.path.join(logdir, f"{alg}_final")
    model.save(save_path)
    env.close()
    eval_env.close()
    return model, f"{save_path}.zip"


@torch.no_grad()
def critic_q_estimate(model, obs: np.ndarray, act: np.ndarray):
    """
    Works with SB3 2.7+ and older:
    - For TD3: returns (q1, q2, qmin)
    - For DDPG: returns q
    """
    device = model.device
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
    act_t = torch.as_tensor(act, dtype=torch.float32, device=device)
    if act_t.ndim == 1:
        act_t = act_t.unsqueeze(-1)

    policy = model.policy

    # === Get the correct feature extractor ===
    # TD3/DDPG in SB3 2.x store extractors inside actor/critic
    features_extractor = getattr(policy, "features_extractor", None)
    if features_extractor is None:
        if hasattr(policy, "actor") and hasattr(policy.actor, "features_extractor"):
            features_extractor = policy.actor.features_extractor
        elif hasattr(model, "critic") and hasattr(model.critic, "features_extractor"):
            features_extractor = model.critic.features_extractor
        else:
            raise RuntimeError(
                "Could not find any features_extractor in policy or critic"
            )

    # === Extract features ===
    # This always works with both SB3 2.7 and older.
    preprocessed_obs = policy.obs_to_tensor(obs_t)[0]  # preprocess via policy
    features = features_extractor(preprocessed_obs)
    if isinstance(features, tuple):
        features = torch.cat([f for f in features if torch.is_tensor(f)], dim=1)

    # === Combine features and actions ===
    q_input = torch.cat([features, act_t], dim=1)

    critic = getattr(model, "critic", None)
    if critic is None:
        raise RuntimeError("Model has no critic module")

    # === TD3: two Q networks ===
    if hasattr(critic, "q_networks"):
        qnets = critic.q_networks
        if len(qnets) == 1:  # DDPG
            q = qnets[0](q_input).squeeze(-1).detach().cpu().numpy()
            return q
        else:  # TD3
            q1 = qnets[0](q_input).squeeze(-1).detach().cpu().numpy()
            q2 = qnets[1](q_input).squeeze(-1).detach().cpu().numpy()
            return q1, q2, np.minimum(q1, q2)

    # === Older SB3 fallback ===
    if hasattr(critic, "qf1") and hasattr(critic, "qf2"):
        q1 = critic.qf1(q_input).squeeze(-1).detach().cpu().numpy()
        q2 = critic.qf2(q_input).squeeze(-1).detach().cpu().numpy()
        return q1, q2, np.minimum(q1, q2)
    if hasattr(policy, "q_net"):
        q = policy.q_net(q_input).squeeze(-1).detach().cpu().numpy()
        return q

    raise RuntimeError("Could not access critics for this SB3 version (TD3/DDPG).")


def to_pyint(x):
    """Convert numpy scalars / python ints to a built-in int."""
    # handles np.int64, np.int32, numpy scalar arrays, etc.
    return int(np.asarray(x).item())


def rollout_and_bias(model, episodes=10, gamma=GAMMA, seed=SEED + 123):
    """Run deterministic rollouts, collect (obs, action, return-to-go) and compare with critic estimates."""
    episodes = to_pyint(episodes)
    seed = to_pyint(seed)

    env = gym.make(ENV_ID)
    env.reset(seed=seed)

    rng = np.random.default_rng(seed)
    returns = []
    all_obs, all_act, all_rtgs = [], [], []

    for _ in range(episodes):
        ep_seed = to_pyint(rng.integers(0, 10_000, dtype=np.int64))  # <<< key line
        obs, info = env.reset(seed=ep_seed)

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

        # MC return-to-go
        G = 0.0
        rtgs = []
        for r in reversed(traj_rew):
            G = r + gamma * G
            rtgs.append(G)
        rtgs.reverse()

        returns.append(sum(traj_rew))
        all_obs.extend(traj_obs)
        all_act.extend(traj_act)
        all_rtgs.extend(rtgs)

    env.close()

    all_obs = np.array(all_obs, dtype=np.float32)
    all_act = np.array(all_act, dtype=np.float32)
    all_rtgs = np.array(all_rtgs, dtype=np.float32)

    q_est = critic_q_estimate(model, all_obs, all_act)
    if isinstance(q_est, tuple):
        q1, q2, qmin = q_est
        return {
            "episode_returns": returns,
            "mean_return": float(np.mean(returns)),
            "std_return": float(np.std(returns)),
            "bias_q1": float(np.mean(q1 - all_rtgs)),
            "bias_q2": float(np.mean(q2 - all_rtgs)),
            "bias_qmin": float(np.mean(qmin - all_rtgs)),
        }
    else:
        q = q_est
        return {
            "episode_returns": returns,
            "mean_return": float(np.mean(returns)),
            "std_return": float(np.std(returns)),
            "bias": float(np.mean(q - all_rtgs)),
        }
