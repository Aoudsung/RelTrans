import argparse
import os

import csv
from datetime import datetime
from tqdm import trange
import numpy as np
import gym

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from policy import policy_func
from decision_transformer.Teacher import DC
from decision_transformer.losses import Soft_Distillation_Loss

from decision_transformer.utils import evaluate_on_env, \
    get_d4rl_normalized_score, TrajectoryDataset

import wandb

seed_value = 1234


def train(args):
    print(args.use_residual)
    envname = args.env
    dataname = args.dataset
    biasinfo = args.bias
    env_info = envname + '_' + dataname + '_' + str(biasinfo)

    wandb.init(project="GPRT",
               entity="aoudsung",
               name=f"",
               config={
                   "env_name": 'Temperature-4.0',
                   "latent_dim": args.laten_dim,
                   "bias": args.bias,
                   "dr": args.dr,
               })
    torch.set_num_threads(1)
    dataset = args.dataset  # medium / medium-replay / medium-expert
    rtg_scale = args.rtg_scale  # normalize returns to go
    # use v3 env for evaluation because
    # Decision Transformer paper evaluates results on v3 envs

    if args.env == 'walker2d':
        env_name = 'Walker2d-v3'
        rtg_target = None
        env_d4rl_name = f'walker2d-{dataset}-v2'
        dataset_name = f'walker2d-{dataset}'
    elif args.env == 'halfcheetah':
        env_name = 'HalfCheetah-v3'
        rtg_target = None
        env_d4rl_name = f'halfcheetah-{dataset}-v2'
        dataset_name = f'halfcheetah-{dataset}'
    elif args.env == 'hopper':
        env_name = 'Hopper-v3'
        rtg_target = None
        env_d4rl_name = f'hopper-{dataset}-v2'
        dataset_name = f'hopper-{dataset}'

    else:
        raise NotImplementedError

    max_eval_ep_len = args.max_eval_ep_len  # max len of one episode
    num_eval_ep = args.num_eval_ep  # num of evaluation episodes

    batch_size = args.batch_size  # training batch size
    lr = args.lr  # learning rate
    dr = args.dr  # disstill rate
    wt_decay = args.wt_decay  # weight decay
    warmup_steps = args.warmup_steps  # warmup steps for lr scheduler

    # total updates = max_train_iters x num_updates_per_iter
    max_train_iters = args.max_train_iters
    num_updates_per_iter = args.num_updates_per_iter

    context_len = args.context_len  # K in decision transformer
    n_blocks = args.n_blocks  # num of transformer blocks
    embed_dim = args.embed_dim  # embedding (hidden) dim of transformer
    n_heads = args.n_heads  # num of transformer heads
    dropout_p = args.dropout_p  # dropout probability
    laten_dim = args.laten_dim
    use_residual = args.use_residual
    # load data from this file

    dataset_path = f'{args.dataset_dir}/{dataset_name}.pkl'

    # saves model and csv in this directory
    log_dir = f'BestModel_R/{args.env}/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # training and evaluation device
    device = torch.device(args.device)
    torch.manual_seed(seed_value)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed_value)
        cudnn.deterministic = False
        cudnn.benchmark = True
    np.random.seed(seed_value)

    start_time = datetime.now().replace(microsecond=0)
    start_time_str = start_time.strftime("%y-%m-%d-%H-%M-%S")

    prefix = "dt_" + env_d4rl_name

    save_model_name = prefix + "_model_" + start_time_str + ".pt"
    save_model_path = os.path.join(log_dir, save_model_name)
    save_best_model_path = save_model_path[:-3] + "_best.pt"

    log_csv_name = prefix + "_log_" + start_time_str + ".csv"
    log_csv_path = os.path.join(log_dir, log_csv_name)

    csv_writer = csv.writer(open(log_csv_path, 'a', 1))
    csv_header = (["duration", "num_updates", "action_loss",
                   "eval_avg_reward", "eval_avg_ep_len", "eval_d4rl_score"])

    csv_writer.writerow(csv_header)

    print("=" * 60)
    print("start time: " + start_time_str)
    print("=" * 60)

    print("device set to: " + str(device))
    print("dataset path: " + dataset_path)
    print("model save path: " + save_model_path)
    print("log csv save path: " + log_csv_path)

    traj_dataset = TrajectoryDataset(dataset_path, context_len, rtg_scale)

    ## get state stats from dataset
    state_mean, state_std = traj_dataset.get_state_stats()

    env = gym.make(env_name)

    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    Policy = policy_func(
        state_dim=state_dim,
        act_dim=act_dim,
        h_dim=embed_dim * 4,
        context_len=context_len,
        lanten_dim=laten_dim,
        residual=use_residual
    ).to(device)

    critic = DC(
        state_dim=state_dim,
        act_dim=act_dim,
        n_blocks=n_blocks,
        h_dim=embed_dim,
        context_len=context_len,
        n_heads=n_heads,
        drop_p=dropout_p, ).to(device)

    SoftLoss = Soft_Distillation_Loss(
        lambda_balancing=args.bias,
    ).to(device)

    optimizer = torch.optim.AdamW(
        [{'params': Policy.parameters()}, {'params': critic.parameters()}],
        lr=lr,
        weight_decay=wt_decay,
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: min((steps + 1) / warmup_steps, 1)
    )

    max_d4rl_score = -1.0
    total_updates = 0
    eval_avg_ep_len = 0
    d4rl_score = []
    for _ in trange(max_train_iters):

        log_action_losses = []
        Policy.train()
        critic.train()
        SoftLoss.train()

        now = max(100, int(eval_avg_ep_len))
        for _ in range(num_updates_per_iter):
            timesteps, states, actions, next_states, returns_to_go, traj_mask, rewards = traj_dataset.Get_batch(
                batch_size, now)
            timesteps = timesteps.to(device)  # B x T
            states = states.to(device)  # B x T x state_dim
            rewards = rewards.to(device)  # B x T
            returns_to_go = returns_to_go.to(device).unsqueeze(dim=-1)  # B x T
            actions = actions.to(device)  # B x T x act_dim
            action_target = torch.clone(actions).detach().to(device)

            action_pred, _, _ = critic.forward(timesteps, states, actions, rewards.unsqueeze(-1))
            # action_pred, _, _ = critic.forward(timesteps, states, actions, returns_to_go)

            a_pred = Policy.forward(
                states=states,
                timesteps=timesteps
            )

            # action_pred = action_pred.view(-1, act_dim)[traj_mask.view(-1, ) > 0]
            # action_target = action_target.view(-1, act_dim)[traj_mask.view(-1, ) > 0]
            # a_pred = a_pred.view(-1, act_dim)[traj_mask.view(-1, ) > 0]

            action_loss = SoftLoss.forward(action_pred, a_pred, action_target, dr)

            optimizer.zero_grad()
            action_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.25)
            torch.nn.utils.clip_grad_norm_(Policy.parameters(), 0.25)
            optimizer.step()
            scheduler.step()

            log_action_losses.append(action_loss.detach().cpu().item())

        results = evaluate_on_env(Policy, device, context_len, env, rtg_target, rtg_scale,
                                  num_eval_ep, max_eval_ep_len, state_mean, state_std)
        eval_avg_reward = results['eval/avg_reward']
        eval_avg_ep_len = results['eval/avg_ep_len']
        eval_d4rl_score = get_d4rl_normalized_score(results['eval/avg_reward'], env_name) * 100
        d4rl_score.append(eval_d4rl_score)
        mean_action_loss = np.mean(log_action_losses)
        time_elapsed = str(datetime.now().replace(microsecond=0) - start_time)

        total_updates += num_updates_per_iter

        log_str = ("=" * 60 + '\n' +
                   "time elapsed: " + time_elapsed + '\n' +
                   "num of updates: " + str(total_updates) + '\n' +
                   "action loss: " + format(mean_action_loss, ".5f") + '\n' +
                   "eval avg reward: " + format(eval_avg_reward, ".5f") + '\n' +
                   "eval avg ep len: " + format(eval_avg_ep_len, ".5f") + '\n' +
                   "eval d4rl score: " + format(eval_d4rl_score, ".5f")
                   )
        wandb.log({
            'return_mean': eval_avg_reward,
            'd4rl_score': eval_d4rl_score,
        }, step=total_updates)
        print(log_str)

        log_data = [time_elapsed, total_updates, mean_action_loss,
                    eval_avg_reward, eval_avg_ep_len,
                    eval_d4rl_score]

        csv_writer.writerow(log_data)

        # save model
        print("max d4rl score: " + format(max_d4rl_score, ".5f"))
        if eval_d4rl_score >= max_d4rl_score:
            s = {'actor': Policy.state_dict(),
                 'critic': critic.state_dict()}
            torch.save(s, save_best_model_path)

            max_d4rl_score = eval_d4rl_score

    log_path = log_dir + prefix + '.txt'
    with open(log_path, 'a') as f:
        f.write(save_model_name + "-—-—-current parameter is :" + str(laten_dim) + "***" + str(
            use_residual) + "****" + '--num_updates_per_iter:' + str(num_updates_per_iter) + "----" + str(dr) + '\n')
        f.write(str(sum(d4rl_score) / len(d4rl_score)) + '\n')
    print("=" * 60)
    print("finished training!")
    print("=" * 60)
    end_time = datetime.now().replace(microsecond=0)
    time_elapsed = str(end_time - start_time)
    end_time_str = end_time.strftime("%y-%m-%d-%H-%M-%S")
    print("started training at: " + start_time_str)
    print("finished training at: " + end_time_str)
    print("total training time: " + time_elapsed)
    print("max d4rl score: " + format(max_d4rl_score, ".5f"))
    print("saved max d4rl score model at: " + save_best_model_path)
    print("saved last updated model at: " + save_model_path)
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # parser.add_argument('--env', type=str, default='hopper')
    parser.add_argument('--env', type=str, default='halfcheetah')
    # parser.add_argument('--env', type=str, default='walker2d')
    # parser.add_argument('--dataset', type=str, default='medium-v2')
    parser.add_argument('--dataset', type=str, default='medium-expert-v2')
    # parser.add_argument('--dataset', type=str, default='expert-v2')
    # parser.add_argument('--dataset', type=str, default='medium-replay-v2')
    parser.add_argument('--rtg_scale', type=int, default=1000)

    parser.add_argument('--max_eval_ep_len', type=int, default=1000)
    parser.add_argument('--num_eval_ep', type=int, default=10)

    parser.add_argument('--dataset_dir', type=str, default='data/')
    # parser.add_argument('--log_dir', type=str, default='BestModel_R/halfcheetah/')
    parser.add_argument('--log_dir', type=str, default='BestModel_R/ant/')

    parser.add_argument('--context_len', type=int, default=32)
    parser.add_argument('--n_blocks', type=int, default=3)
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--n_heads', type=int, default=1)
    parser.add_argument('--dropout_p', type=float, default=0.1)
    parser.add_argument('--laten_dim', type=int, default=128)
    parser.add_argument('--use_residual', type=bool, default=False)

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--dr', type=float, default=6.0)
    parser.add_argument('--bias', type=float, default=0.8)
    parser.add_argument('--wt_decay', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=10000)

    parser.add_argument('--max_train_iters', type=int, default=300)
    parser.add_argument('--num_updates_per_iter', type=int, default=100)

    parser.add_argument('--device', type=str, default='cuda:1')
    args = parser.parse_args()

    train(args)
