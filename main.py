import copy
import glob
import os
import time
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.algo import gail
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from evaluation import evaluate


def main():
    args = get_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    import wandb
    if args.wandb:
        wandb.login()
        wandb.init(project="img-gen-rl", entity="rowing0914", name="gail", group="gail", dir="/tmp/wandb")
        wandb.config.update(args)

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    # ================= mocap
    from dm_control.suite import humanoid_CMU
    import sys
    sys.path.append("../merel-mocap-gail-modified/")
    from mocap_gail.mujoco_dataset import Mujoco_Dset

    args.obs_only = True
    env = humanoid_CMU.run()
    
    from run_collect_cmu_mocap import get_humanoid_cmu_obs
    obs = get_humanoid_cmu_obs(env)
    env.task.random.seed(args.seed)
    dataset = Mujoco_Dset(expert_path=args.mocap_expert_path, traj_limitation=args.mocap_traj_limitation, obs_only=args.obs_only)
    # ob_expert, ac_or_qpos_expert = dataset.get_next_batch(32)
    ob_expert, ac_or_qpos_expert = dataset.obs[0], dataset.qpos[0]
    print(ob_expert.shape, ac_or_qpos_expert.shape)

    # # Set the Humanoid to the init position
    qpos = ac_or_qpos_expert[0]  # qpos at t = 0
    with env.physics.reset_context():
        env.physics.data.qpos[:] = qpos[:63]
        env.physics.data.qvel[:] = qpos[63:]
    # ================= mocap
    envs = env
    
    # envs = make_vec_envs(args.env_name, args.seed, args.num_processes, args.gamma, args.log_dir, device, False)
    # obs_shape = envs.observation_space.shape
    # act_sp_type = envs.action_space.__class__.__name__
    # act_shape = envs.action_space.shape
    obs_shape = obs.shape
    act_sp_type = "Box"
    ac_spec = env.action_spec()
    act_shape = ac_spec.shape[0]
    actor_critic = Policy(obs_shape, act_sp_type, act_shape, base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic.to(device)

    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(
            actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo':
        agent = algo.PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(actor_critic, args.value_loss_coef, args.entropy_coef, acktr=True)

    if args.gail:
        assert len(obs.shape) == 1
        discr = gail.Discriminator(obs.shape[0] + envs.action_space.shape[0], 100, device)
        file_name = os.path.join(args.gail_experts_dir, "trajs_{}.pt".format(args.env_name.split('-')[0].lower()))
        
        expert_dataset = gail.ExpertDataset(file_name, num_trajectories=4, subsample_frequency=20)
        drop_last = len(expert_dataset) > args.gail_batch_size
        gail_train_loader = torch.utils.data.DataLoader(
            dataset=expert_dataset,
            batch_size=args.gail_batch_size,
            shuffle=True,
            drop_last=drop_last)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)

    start = time.time()
    num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes
    
    # Init eval w/h random agent
    vec_norm = utils.get_vec_normalize(envs)
    obs_rms = vec_norm.obs_rms if vec_norm is not None else None
    eval_return = evaluate(actor_critic, obs_rms, args.env_name, args.seed, args.num_processes, eval_log_dir, device)
    if args.wandb:
        wandb.log(data={"eval/ep_return": eval_return}, step=0)
    
    for j in range(num_updates):
        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(agent.optimizer, j, num_updates, agent.optimizer.lr if args.algo == "acktr" else args.lr)

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor([[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action, action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(rollouts.obs[-1], rollouts.recurrent_hidden_states[-1], rollouts.masks[-1]).detach()

        if args.gail:
            # if j >= 10:
            #     envs.venv.eval()

            gail_epoch = args.gail_epoch
            if j < 10:
                gail_epoch = 100  # Warm up
            for _ in range(gail_epoch):
                vec_norm = utils.get_vec_normalize(envs)
                obfilt = vec_norm._obfilt if vec_norm is not None else None
                discr.update(gail_train_loader, rollouts, obfilt)

            for step in range(args.num_steps):
                rollouts.rewards[step] = discr.predict_reward(rollouts.obs[step], rollouts.actions[step], args.gamma, rollouts.masks[step])

        rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.gae_lambda, args.use_proper_time_limits)
        value_loss, action_loss, dist_entropy = agent.update(rollouts)
        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0 or j == num_updates - 1) and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            torch.save([actor_critic, getattr(utils.get_vec_normalize(envs), 'obs_rms', None)], os.path.join(save_path, args.env_name + ".pt"))

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            print(f"[Train] TS: {total_num_steps} Ep-reward {np.mean(episode_rewards):.2f}, V-loss: {value_loss}, Pi-loss: {action_loss}")
            if args.wandb:
                wandb.log(data={"train/ep_return": np.mean(episode_rewards)}, step=total_num_steps)

        if (args.eval_interval is not None and len(episode_rewards) > 1 and j % args.eval_interval == 0):
            vec_norm = utils.get_vec_normalize(envs)
            obs_rms = vec_norm.obs_rms if vec_norm is not None else None
            evaluate(actor_critic, obs_rms, args.env_name, args.seed, args.num_processes, eval_log_dir, device)


if __name__ == "__main__":
    main()
