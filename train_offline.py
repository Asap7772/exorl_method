import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)

import os

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

from pathlib import Path
import numpy as np
import torch
from dm_env import specs

import dmc
import utils
from logger import Logger
from replay_buffer import make_replay_loader
from video import VideoRecorder

torch.backends.cudnn.benchmark = True
import wandb
import argparse

def get_domain(task):
    if task.startswith('point_mass_maze'):
        return 'point_mass_maze'
    return task.split('_', 1)[0]


def get_data_seed(seed, num_data_seeds):
    return (seed - 1) % num_data_seeds + 1


def eval(global_step, agent, env, logger, num_eval_episodes, video_recorder):
    step, episode, total_reward = 0, 0, 0
    eval_until_episode = utils.Until(num_eval_episodes)
    while eval_until_episode(episode):
        time_step = env.reset()
        video_recorder.init(env, enabled=(episode == 0))
        while not time_step.last():
            with torch.no_grad(), utils.eval_mode(agent):
                action = agent.act(time_step.observation,
                                   global_step,
                                   eval_mode=True)
            time_step = env.step(action)
            video_recorder.record(env)
            total_reward += time_step.reward
            step += 1

        episode += 1
        video_recorder.save(f'{global_step}.mp4')

    eval_stats = {
        'total_reward': total_reward,
        'num_steps': step,
        'num_episodes': episode,
        'episode_reward': total_reward / episode,
        'episode_length': step / episode,
        'step': global_step
    }

    wandb.log(eval_stats, step=global_step)


    with logger.log_and_dump_ctx(global_step, ty='eval') as log:
        log('episode_reward', total_reward / episode)
        log('episode_length', step / episode)
        log('step', global_step)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb_project', type=str, default='exorl_cql')
    parser.add_argument('--name', type=str, default='cql')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--critic_target_tau', type=float, default=0.01)
    parser.add_argument('--agent', type=object, default=agent.cql.CQLAgent)
    parser.add_argument('--n_samples', type=int, default=3)
    parser.add_argument('--use_critic_lagrange', action='store_true')
    parser.add_argument('--alpha', type=float, default=0.01)
    parser.add_argument('--target_cql_penalty', type=float, default=5.0)
    parser.add_argument('--use_tb', action='store_false')
    parser.add_argument('--hidden_dim', type=int, default=1024)
    parser.add_argument('--nstep', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--has_next_action', action='store_true')
    parser.add_argument('--method', action='store_false')
    parser.add_argument('--method_type', type=int, default=0)
    parser.add_argument('--method_temp', type=float, default=1.0)
    parser.add_argument('--method_alpha', type=float, default=0.5)
    parser.add_argument('--task', type=str, default='walker_walk')
    parser.add_argument('--save_video', action='store_false')
    parser.add_argument('--expl_agent', type=str, default='proto')
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--num_grad_steps', type=int, default=500000)
    parser.add_argument('--log_every_steps', type=int, default=1000)
    parser.add_argument('--num_eval_episodes', type=int, default=10)
    parser.add_argument('--eval_every_steps', type=int, default=10000)
    parser.add_argument('--replay_buffer_dir', type=str, default='../../../datasets')
    parser.add_argument('--replay_buffer_size', type=int, default=1000000)
    parser.add_argument('--replay_buffer_num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=1)
    cfg = parser.parse_args()

    work_dir = Path.cwd()
    print(f'workspace: {work_dir}')

    utils.set_seed_everywhere(cfg.seed)
    device = torch.device(cfg.device)

    # create logger
    logger = Logger(work_dir, use_tb=cfg.use_tb)

    # create envs
    env = dmc.make(cfg.task, seed=cfg.seed)

    # create agent
    agent = cfg.agent(obs_shape=env.observation_spec().shape, action_shape=env.action_spec().shape, **vars(cfg))

    # create replay buffer
    data_specs = (env.observation_spec(), env.action_spec(), env.reward_spec(),
                  env.discount_spec())
    wandb.init(project=cfg.wandb_project, settings=wandb.Settings(start_method="fork"))

    # create data storage
    domain = get_domain(cfg.task)
    datasets_dir = work_dir / cfg.replay_buffer_dir
    replay_dir = datasets_dir.resolve() / domain / cfg.expl_agent / 'buffer'
    print(f'replay dir: {replay_dir}')

    replay_loader = make_replay_loader(env, replay_dir, cfg.replay_buffer_size,
                                       cfg.batch_size,
                                       cfg.replay_buffer_num_workers,
                                       cfg.discount)
    replay_iter = iter(replay_loader)

    # create video recorders
    video_recorder = VideoRecorder(work_dir if cfg.save_video else None)

    timer = utils.Timer()

    global_step = 0

    train_until_step = utils.Until(cfg.num_grad_steps)
    eval_every_step = utils.Every(cfg.eval_every_steps)
    log_every_step = utils.Every(cfg.log_every_steps)

    while train_until_step(global_step):
        # try to evaluate
        if eval_every_step(global_step):
            logger.log('eval_total_time', timer.total_time(), global_step)
            eval(global_step, agent, env, logger, cfg.num_eval_episodes,
                 video_recorder)

        metrics = agent.update(replay_iter, global_step)
        logger.log_metrics(metrics, global_step, ty='train')
        if log_every_step(global_step):
            elapsed_time, total_time = timer.reset()
            with logger.log_and_dump_ctx(global_step, ty='train') as log:
                log('fps', cfg.log_every_steps / elapsed_time)
                log('total_time', total_time)
                log('step', global_step)

        global_step += 1
