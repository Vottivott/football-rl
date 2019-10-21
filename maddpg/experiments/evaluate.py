import argparse
import numpy as np
import tensorflow as tf
import time
import pickle

import maddpg.common.tf_util as U
from maddpg.trainer.maddpg import MADDPGAgentTrainer
from maddpg.trainer.maddpg import update_fast
import tensorflow.contrib.layers as layers

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=100, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=60000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024*32, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=256, help="number of units in the mlp") #64
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default=None, help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="/tmp/policy/", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=1000, help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="", help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/", help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/", help="directory where plot data is saved")
    return parser.parse_args()

def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    # Note(Daniel): No it fucking doesn't. It sometimes does that and sometimes it return just one value. I don't understand why.
    #               ie. in one place (p_train) it's defined as model(in, num_actions),
    #                   but twice (p_train & q_train) it's defined as model(in, 1). wtf? 
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out

def make_env(scenario_name, arglist, benchmark=False):
    from multiagent.environment import MultiAgentEnv, BatchMultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    def done_callback(agent, world):
        if hasattr(world, 'is_scenareo_over'):
            return world.is_scenareo_over(agent, world)
        return False

    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data, done_callback = done_callback)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, done_callback = done_callback)
        #env = BatchMultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env

def get_trainers(env, num_adversaries, obs_shape_n, arglist):
    trainers = []
    model = mlp_model
    trainer = MADDPGAgentTrainer
    for i in range(num_adversaries):
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.adv_policy=='ddpg')))
    for i in range(num_adversaries, env.n):
        print(i)
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.good_policy=='ddpg')))
    return trainers

import os
def eval(arglist):
    if False:
        graphs = [tf.Graph()]
        with graphs[0].as_default():
            sessions = [U.single_threaded_session()]
            with sessions[0].as_default():
                # Create environment
                env = make_env(arglist.scenario, arglist, arglist.benchmark)
                # Create agent trainers
                obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
                num_adversaries = 0
                trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)
                trainers_list = [trainers]
                U.initialize()
                U.load_state(arglist.load_dir+ '#'+"{:06d}".format(60))

    else:
        graphs = []
        sessions = []
        trainers_list = []
        num_adversaries = 0
        env = make_env(arglist.scenario, arglist, arglist.benchmark)
        for train_time in range(1, 60, 10):
            
            g = tf.Graph()
            with g.as_default():
                s = tf.Session()
                with s.as_default():
                    try:
                        path = arglist.load_dir+ '#'+"{:06d}".format(train_time)
                        if os.path.isfile(path+'.index'):
                            print("setting up session", train_time)
                            obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
                            trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)
                            trainers_list.append(trainers)
                            U.initialize()
                            U.load_state(path)
                            graphs.append(g)
                            sessions.append(s)
                        else:
                            print('File is missing! Ignoring it.')
                    except:
                        print('missing at. Pretending like it never happened...',train_time)

    print("NUMBER OF SESSIONS TO COMPARE:",len(sessions))
    num_its = 50
    results = np.zeros((len(sessions),len(sessions)))
    obs_n = env.reset()
    team_size = len(obs_n)//2
    episode_step = 0
    for ia, a in enumerate(sessions):
        for ib, b in enumerate(sessions):
            if ia > ib: 
                continue
            #try:
            for i in range(num_its):
                while True:
                    with graphs[ia].as_default():
                        with a.as_default():
                            action_a = trainers_list[ia][0].act(obs_n[:team_size])
                    with graphs[ib].as_default():
                        with b.as_default():
                            action_b = trainers_list[ib][0].act(obs_n[team_size:])

                    action_n = np.concatenate((action_a, action_b))


                    obs_n, rew_n, done_n, info_n = env.step(action_n)
                    episode_step += 1
                    done = all(done_n)
                    terminal = (episode_step >= arglist.max_episode_len)
                    
                    #time.sleep(0.08)
                    #env.render()

                    if done or terminal:
                        episode_step = 0
                        if not done:
                            results[ia][ib] += 0.5
                        else: 
                            if env.world.landmarks[0].state.p_pos[0] > 0:
                                results[ia][ib] += 1.0
                        obs_n = env.reset()
                        break
            #except:
                #print('Failed evaluation... Did we load things right?')
            print('Compared', ia, 'aginst', ib, 'got:', results[ia][ib])

    print(results / num_its)
    np.save(arglist.load_dir+"_resutls", results/num_its)


if __name__ == '__main__':
    arglist = parse_args()
    eval(arglist)
