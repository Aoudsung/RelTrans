import random
import time
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset
from decision_transformer.d4rl_infos import REF_MIN_SCORE, REF_MAX_SCORE, D4RL_DATASET_STATS


def discount_cumsum(x, gamma):
    disc_cumsum = np.zeros_like(x)
    disc_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        disc_cumsum[t] = x[t] + gamma * disc_cumsum[t+1]
    return disc_cumsum


def get_d4rl_normalized_score(score, env_name):
    env_key = env_name.split('-')[0].lower()
    assert env_key in REF_MAX_SCORE, f'no reference score for {env_key} env to calculate d4rl score'
    return (score - REF_MIN_SCORE[env_key]) / (REF_MAX_SCORE[env_key] - REF_MIN_SCORE[env_key])


def get_d4rl_dataset_stats(env_d4rl_name):
    return D4RL_DATASET_STATS[env_d4rl_name]


def evaluate_on_env(model, device, context_len, env, rtg_target, rtg_scale,
                    num_eval_ep=10, max_test_ep_len=1000,
                    state_mean=None, state_std=None, render=False):


    eval_batch_size = 1  # required for forward pass

    results = {}
    total_reward = 0
    total_timesteps = 0

    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    if state_mean is None:
        state_mean = torch.zeros((state_dim,)).to(device)
    else:
        state_mean = torch.from_numpy(state_mean).to(device)

    if state_std is None:
        state_std = torch.ones((state_dim,)).to(device)
    else:
        state_std = torch.from_numpy(state_std).to(device)

    # same as timesteps used for training the transformer
    # also, crashes if device is passed to arange()
    timesteps = torch.arange(start=0, end=max_test_ep_len, step=1)
    timesteps = timesteps.repeat(eval_batch_size, 1).to(device)

    model.eval()

    with torch.no_grad():

        for _ in range(num_eval_ep):

            # zeros place holders
            actions = torch.zeros((eval_batch_size, max_test_ep_len, act_dim),
                                dtype=torch.float32, device=device)
            states = torch.zeros((eval_batch_size, max_test_ep_len, state_dim),
                                dtype=torch.float32, device=device)
            rewards = torch.zeros((eval_batch_size, max_test_ep_len, 1),
                                dtype=torch.float32, device=device)
            rewards_to_go = rewards = torch.zeros((eval_batch_size, max_test_ep_len, 1),
                                dtype=torch.float32, device=device)
            # init episode
            running_state = env.reset()
            running_reward = 0
            running_rtg = rtg_target / rtg_scale

            for t in range(max_test_ep_len):

                total_timesteps += 1

                # add state in placeholder and normalize
                states[0, t] = torch.from_numpy(running_state).to(device)
                states[0, t] = (states[0, t] - state_mean) / state_std

                rewards[0, t] = running_reward

                # calcualate running rtg and add it in placeholder
                running_rtg = running_rtg - (running_reward / rtg_scale)
                rewards_to_go[0, t] = running_rtg

                if t < context_len:

                    act_preds = model.forward(  timesteps[:,:context_len],
                                                states[:,:context_len],
                                            )
                    # print(act_preds.shape)
                    act = act_preds[0, t].detach()
                else:

                    act_preds = model.forward( timesteps[:,t-context_len+1:t+1],
                                                states[:,t-context_len+1:t+1],
                                            )
                    act = act_preds[0, -1].detach()

                running_state, running_reward, done, _ = env.step(act.cpu().numpy())
                # add action in placeholder
                actions[0, t] = act

                total_reward += running_reward

                if render:
                    env.render()
                if done:
                    break

    results['eval/avg_reward'] = total_reward / num_eval_ep
    results['eval/avg_ep_len'] = total_timesteps / num_eval_ep

    return results


class D4RLTrajectoryDataset(Dataset):
    def __init__(self, dataset_path, context_len, rtg_scale):

        self.context_len = context_len

        # load dataset
        with open(dataset_path, 'rb') as f:
            self.trajectories = pickle.load(f)

        # calculate min len of traj, state mean and variance
        # and returns_to_go for all traj
        min_len = 10**6
        states = []
        next_states = []
        rtg = []

        for traj in self.trajectories:
            traj_len = traj['observations'].shape[0]
            min_len = min(min_len, traj_len)
            states.append(traj['observations'])
            next_states.append(traj['next_observations'])

            # calculate returns to go and rescale them
            traj['returns_to_go'] = discount_cumsum(traj['rewards'], 1.0) / rtg_scale
            print(traj['returns_to_go'])

        # used for input normalization
        states = np.concatenate(states, axis=0)
        self.state_mean, self.state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
        next_states = np.concatenate(next_states, axis=0)
        self.next_state_mean, self.next_state_std = np.mean(next_states, axis=0), np.std(next_states, axis=0) + 1e-6

    def get_state_stats(self):
        return self.state_mean, self.state_std

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):

        traj = self.trajectories[idx]
        traj_len = traj['observations'].shape[0]

        if traj_len - 1 >= self.context_len:
            # sample random index to slice trajectory
            si = random.randint(0, traj_len - self.context_len - 1)
            t = si
            ti = random.randint(1, 100)
            a = si % ti
            b = si // ti
            si = a * 10 + b
            if si > traj_len - self.context_len - 1:
                si = t
            states = torch.from_numpy(traj['observations'][si : si + self.context_len])
            next_states = torch.from_numpy(traj['next_observations'][si : si + self.context_len])
            actions = torch.from_numpy(traj['actions'][si : si + self.context_len])
            returns_to_go = torch.from_numpy(traj['returns_to_go'][si : si + self.context_len])
            rewards = torch.from_numpy(traj['rewards'][si: si+self.context_len])
            timesteps = torch.arange(start=si, end=si+self.context_len, step=1)
            next_rewards = torch.from_numpy(traj['rewards'][si+1: si+self.context_len+1])
            # all ones since no padding
            traj_mask = torch.ones(self.context_len)

        else:
            padding_len = self.context_len - traj_len

            # padding with zeros
            states = torch.from_numpy(traj['observations'])

            states = torch.cat([states,
                                torch.zeros(([padding_len] + list(states.shape[1:])),
                                dtype=states.dtype)],
                               dim=0)
            next_states = torch.from_numpy(traj['next_observations'])
            next_states = torch.cat([next_states,
                                torch.zeros(([padding_len] + list(states.shape[1:])),
                                dtype=states.dtype)],
                               dim=0)
            actions = torch.from_numpy(traj['actions'])
            actions = torch.cat([actions,
                                torch.zeros(([padding_len] + list(actions.shape[1:])),
                                dtype=actions.dtype)],
                               dim=0)

            returns_to_go = torch.from_numpy(traj['returns_to_go'])
            returns_to_go = torch.cat([returns_to_go,
                                torch.zeros(([padding_len] + list(returns_to_go.shape[1:])),
                                dtype=returns_to_go.dtype)],
                               dim=0)
            rewards = torch.from_numpy(traj['rewards'])
            rewards = torch.cat([rewards,
                                torch.zeros(([padding_len] + list(rewards.shape[1:])),
                                dtype=rewards.dtype)],
                               dim=0)
            next_rewards = torch.from_numpy(traj['rewards'][1:])
            next_rewards = torch.cat([next_rewards,
                                torch.zeros(([padding_len+1] + list(next_rewards.shape[1:])),
                                dtype=next_rewards.dtype)],
                               dim=0)
            timesteps = torch.arange(start=0, end=self.context_len, step=1)

            traj_mask = torch.cat([torch.ones(traj_len),
                                   torch.zeros(padding_len)],
                                  dim=0)

        return  timesteps, states, actions, next_states, returns_to_go, rewards, next_rewards,traj_mask

class TrajectoryDataset():
    def __init__(self, dataset_path, context_len, rtg_scale):

        self.context_len = context_len

        # load dataset
        with open(dataset_path, 'rb') as f:
            self.trajectories = pickle.load(f)

        # calculate min len of traj, state mean and variance
        # and returns_to_go for all traj
        min_len = 10 ** 6
        states = []
        next_states = []
        rtg = []
        rwd = []
        for traj in self.trajectories:
            traj_len = traj['observations'].shape[0]
            min_len = min(min_len, traj_len)
            states.append(traj['observations'])
            next_states.append(traj['next_observations'])
            rwd.append(traj['rewards'].sum())
            # calculate returns to go and rescale them
            traj['returns_to_go'] = discount_cumsum(traj['rewards'], 0.9) / rtg_scale
            rtg.append(discount_cumsum(traj['rewards'], 1.0) / rtg_scale)
        # used for input normalization
        states = np.concatenate(states, axis=0)
        self.state_mean, self.state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
        next_states = np.concatenate(next_states, axis=0)
        self.next_state_mean, self.next_state_std = np.mean(next_states, axis=0), np.std(next_states, axis=0) + 1e-6
        # rwd = np.array(rwd)
        # mx, mn = max(rwd), min(rwd)
        # norm_rwd = (rwd - mn) / (mx - mn)
        # self.p_sample = norm_rwd / sum(norm_rwd)
        # normalize states
        for traj in self.trajectories:
            traj['observations'] = (traj['observations'] - self.state_mean) / self.state_std
            traj['next_observations'] = (traj['next_observations'] - self.next_state_mean) / self.next_state_std

    def get_state_stats(self):
        return self.state_mean, self.state_std

    def __len__(self):
        return len(self.trajectories)

    def get_item(self, idx, now):

        traj = self.trajectories[idx]
        traj_len = traj['observations'].shape[0]

        if traj_len - 1 >= self.context_len:

            # sample random index to slice trajectory
            #si = random.randint(0, min(int(now*3), traj_len - self.context_len - 1))
            si = random.randint(0, traj_len - self.context_len - 1)
            states = traj['observations'][si: si + self.context_len]
            next_states = traj['next_observations'][si : si + self.context_len]
            rewards = traj['rewards'][si: si + self.context_len]
            actions = traj['actions'][si: si + self.context_len]
            returns_to_go = traj['returns_to_go'][si: si + self.context_len]
            timesteps = np.arange(start=si, stop=si + self.context_len, step=1)
            # all ones since no padding
            traj_mask = np.ones(self.context_len)

        else:
            padding_len = self.context_len - traj_len

            # padding with zeros
            states = traj['observations']

            states = np.concatenate([states,
                                np.zeros(([padding_len] + list(states.shape[1:])),
                                            dtype=states.dtype)],
                               )
            next_states = traj['next_observations']
            next_states = np.concatenate([next_states,
                                np.zeros(([padding_len] + list(next_states.shape[1:])),
                                            dtype=next_states.dtype)],
                               )
            actions = traj['actions']
            actions = np.concatenate([actions,
                                 np.zeros(([padding_len] + list(actions.shape[1:])),
                                             dtype=actions.dtype)],
                                )

            returns_to_go = traj['returns_to_go']
            returns_to_go = np.concatenate([returns_to_go,
                                       np.zeros(([padding_len] + list(returns_to_go.shape[1:])),
                                                   dtype=returns_to_go.dtype)],
                                      )
            rewards = traj['rewards']
            rewards = np.concatenate([rewards,
                                       np.zeros(([padding_len] + list(rewards.shape[1:])),
                                                   dtype=rewards.dtype)],
                                      )
            timesteps = np.arange(start=0, stop=self.context_len, step=1)

            traj_mask = np.concatenate([np.ones(traj_len),
                                   np.zeros(padding_len)],
                                  )

        return timesteps, states, actions, next_states, returns_to_go, traj_mask, rewards

    def Get_batch(self, batch_size, now):
        t, s, a, ns, r, m, re = [], [], [], [], [], [], []
        for i in range(batch_size):
            ti = random.randint(0, len(self.trajectories) - 1)
            timesteps, states, actions, next_states, returns_to_go, traj_mask, rewards = self.get_item(ti, now)
            t = [*t, timesteps]
            s = [*s, states]
            a = [*a, actions]
            ns = [*ns, next_states]
            r = [*r, returns_to_go]
            m = [*m, traj_mask]
            re = [*re, rewards]
        return torch.from_numpy(np.array(t)), torch.from_numpy(np.array(s)), \
            torch.from_numpy(np.array(a)), torch.from_numpy(np.array(ns)), \
            torch.from_numpy(np.array(r)), \
            torch.from_numpy(np.array(m)), torch.from_numpy(np.array(re))
