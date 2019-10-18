import argparse
import numpy as np
import tensorflow as tf
import time
import pickle

import maddpg.common.tf_util as U
from maddpg.trainer.maddpg import MADDPGAgentTrainer
from maddpg.trainer.maddpg import update_fast
import tensorflow.contrib.layers as layers

import video_maker
from folder_tools import clear_folder, rename_single_file_in_folder, read_name_of_single_file_in_folder

from experience_loader import load_new_experiences
from emailer import send_mail_message_with_image, send_mail_message_with_attachment
from plotter import produce_graph

import datetime


        

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="football", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=100, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=1000000000000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=256, help="number of units in the mlp") #64
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default="football", help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="../../policy/", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=1000, help="save model once every time this many episodes are completed") #1000
    parser.add_argument("--load-dir", type=str, default="", help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=True)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/", help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/", help="directory where plot data is saved")

    # Custom arguments
    parser.add_argument("--video", action="store_true", default=False)   
    parser.add_argument("--autoemail", action="store_true", default=False)
    parser.add_argument("--multicomputer-main", action="store_true", default=False)
    parser.add_argument("--multicomputer-worker", action="store_true", default=False)
    parser.add_argument("--team-size", type=int, default=5, help="size of each team")

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
    world = scenario.make_world(arglist.team_size)
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
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.good_policy=='ddpg')))
    return trainers


def train(arglist):
    with U.single_threaded_session():
        # Create environment
        env = make_env(arglist.scenario, arglist, arglist.benchmark)
        # Create agent trainers
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        num_adversaries = min(env.n, arglist.num_adversaries)
        num_adversaries = 0
        trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)
        print('Using good policy {} and adv policy {}'.format(arglist.good_policy, arglist.adv_policy))

        # Initialize
        U.initialize()

        # Load previous results, if necessary
        if arglist.load_dir == "":
            arglist.load_dir = arglist.save_dir
        if arglist.display or arglist.restore or arglist.benchmark:
            print('Loading previous state...')
            U.load_state(arglist.load_dir)

        episode_rewards = [0.0]  # sum of rewards for all agents
        agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
        final_ep_rewards = []  # sum of rewards for training curve
        final_ep_ag_rewards = []  # agent rewards for training curve
        agent_info = [[[]]]  # placeholder for benchmarking info
        saver = tf.train.Saver()
        obs_n = env.reset()
        episode_step = 0
        train_step = 0
        t_start = time.time()

        # Multicomputer stuff
        worker_current_game_experiences = []
        worker_t0 = time.time()

        episode_lengths = [0]
        episode_rewards_smooth = []
        episode_lengths_smooth = []
        episode_rewards_smooth_hi = []
        episode_lengths_smooth_hi = []
        episode_rewards_smooth_lo = []
        episode_lengths_smooth_lo = []

        current_frame = 0

        def save(varname, var):
            fname = varname + ".pkl"
            with open("../../saved_variables/" + fname, 'wb') as fp:
                pickle.dump(var, fp)

        def load_single(varname):
            fname = varname + ".pkl"
            with open("../../saved_variables/" + fname, 'rb') as fp:
                return pickle.load(fp)

        def load(*varnames):
            return tuple(load_single(varname) for varname in varnames)

        print('Starting iterations...')
        single_nn = True
        while True:
            if arglist.multicomputer_main:
                new_experiences = load_new_experiences()
                for exp in new_experiences:
                    obs_n, action_n, rew_n, new_obs_n, done_n, terminal = exp
                    for i, agent in enumerate(trainers):
                        agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i], terminal)
                    for i, rew in enumerate(rew_n):
                        episode_rewards[-1] += abs(rew)
                        #agent_rewards[i][-1] += abs(rew)
                    episode_step += 1
                    if terminal:
                        episode_lengths.append(episode_step)
                        episode_rewards.append(0)
                        #for a in agent_rewards:
                        #    a.append(0)
                        episode_step = 0
                        if len(episode_rewards) % 5000 == 0:
                            save("episode_rewards", episode_rewards)
                            save("episode_lengths", episode_lengths)
                            save("episode_rewards_smooth", episode_rewards_smooth)
                            save("episode_lengths_smooth", episode_lengths_smooth)
                            print("Saved variables")
                            produce_graph(episode_rewards_smooth, episode_lengths_smooth, episode_rewards_smooth_hi, episode_lengths_smooth_hi, episode_rewards_smooth_lo, episode_lengths_smooth_lo, "../../plot.png")
                            print("Plotted data")
                            if len(episode_rewards_smooth) > 0:
                                message = "rew: %.2f    len: %.2f    [%d]" % (episode_rewards_smooth[-1], episode_lengths_smooth[-1], len(episode_rewards))
                            else:
                                message = ""
                            message += " \n"
                            message += " \n" + "hidden layer size: %d" % arglist.num_units
                            message += " \n" + "gamma: %f" % arglist.gamma
                            message += " \n" + "learning rate: %f" % arglist.lr
                            message += " \n" + "max_episode_len: %d" % arglist.max_episode_len
                            send_mail_message_with_image("Football RL", message, "../../plot.png", image_title="Episode %d" % len(episode_rewards))
                            print("Sent mail")
                if len(episode_rewards_smooth) > 0:
                    rename_single_file_in_folder("../../current_episode_num", str(len(episode_rewards)) + " " + str(episode_rewards_smooth[-1]) + " " + str(episode_lengths_smooth[-1]))



                    if len(episode_rewards) % 200 == 0:
                        episode_lengths_smooth.append(sum(episode_lengths[-200:])/200.0)
                        episode_rewards_smooth.append(sum(episode_rewards[-200:])/200.0)
                        episode_lengths_smooth_hi.append(max(episode_lengths[-200:]))
                        episode_rewards_smooth_hi.append(max(episode_rewards[-200:]))
                        episode_lengths_smooth_lo.append(min(episode_lengths[-200:]))
                        episode_rewards_smooth_lo.append(min(episode_rewards[-200:]))


                # update all trainers, if not in display or benchmark mode
                loss = None
                for agent in trainers:
                    agent.preupdate()
                for agent in trainers:
                    loss = agent.update(trainers, train_step)

                U.save_state(arglist.save_dir, saver=saver)
            else:
                # get action
                if single_nn:
                    action_n = [trainers[0].action(obs) for obs in obs_n] # this should be done in parallel!
                else:
                    action_n = [agent.action(obs) for agent, obs in zip(trainers,obs_n)]
                
                # environment step
                new_obs_n, rew_n, done_n, info_n = env.step(action_n)
                episode_step += 1
                done = all(done_n)
                terminal = (episode_step >= arglist.max_episode_len)
                # collect experience
                if arglist.multicomputer_worker:
                    worker_current_game_experiences.append((obs_n, action_n, rew_n, new_obs_n, done_n, terminal))
                else:
                    for i, agent in enumerate(trainers):
                        agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i], terminal)
                
                obs_n = new_obs_n

                for i, rew in enumerate(rew_n):
                    episode_rewards[-1] += abs(rew)
                    agent_rewards[i][-1] += abs(rew)

                if done or terminal:
                    obs_n = env.reset()
                    episode_step = 0
                    episode_rewards.append(0)
                    for a in agent_rewards:
                        a.append(0)
                    agent_info.append([[]])
                    if arglist.multicomputer_worker:
                        if len(episode_rewards) % 200 == 0 and not arglist.display:
                            fname = datetime.datetime.now().strftime('%Y-%m-%d %H.%M.%S.%f') + ".pkl"
                            with open("../../worker_experiences/" + fname, 'wb') as fp:
                                print("\n[%d] Finished 200 games in %.2f seconds" % (len(episode_rewards), time.time() - worker_t0))
                                pickle.dump(worker_current_game_experiences, fp)
                                print("Saved experience file " + fname)
                                print('Loading latest networks...')
                                worker_t0 = time.time()
                                try:
                                    U.load_state(arglist.load_dir)
                                    print("Latest networks loaded in %.2f seconds" % (time.time() - worker_t0))
                                    worker_t0 = time.time()
                                except tf.python.framework.errors_impl.DataLossError:
                                    print("Couldn't read latest network, it's probably being written...")
                            worker_current_game_experiences = []
                        

                # increment global step counter
                train_step += 1

                # for benchmarking learned policies
                if arglist.benchmark:
                    for i, info in enumerate(info_n):
                        agent_info[-1][i].append(info_n['n'])
                    if train_step > arglist.benchmark_iters and (done or terminal):
                        file_name = arglist.benchmark_dir + arglist.exp_name + '.pkl'
                        print('Finished benchmarking, now saving...')
                        with open(file_name, 'wb') as fp:
                            pickle.dump(agent_info[:-1], fp)
                        break
                    continue

            # for displaying learned policies
            if arglist.display:
                time.sleep(0.02)
                env.render()
                if arglist.video:
                    current_frame += 1
                    video_maker.save_frame(current_frame)
                if terminal and len(episode_rewards) % 7 == 0:
                    if arglist.video:
                        video_maker.combine_frames_to_video("../../videos/test_video.mp4")
                        clear_folder("../../frames/")
                        current_frame = 0
                        epnum = read_name_of_single_file_in_folder("../../current_episode_num")
                        send_mail_message_with_attachment("Football RL - Video", "Episode " + epnum, "../../videos/test_video.mp4", image_title="Episode " + epnum)
                        print("Sent video mail. Waiting 10 minutes.")
                        time.sleep(60*10)
                    worker_t0 = time.time()
                    try:
                        U.load_state(arglist.load_dir)
                        print("Latest networks loaded in %.2f seconds" % (time.time() - worker_t0))
                        worker_t0 = time.time()
                    except tf.python.framework.errors_impl.DataLossError:
                        print("Couldn't read latest network, it's probably being written...")
                continue

            # update all trainers, if not in display or benchmark mode
            if not arglist.multicomputer_worker:
                loss = None
                if single_nn and False: #minor speedup and not working at all...
                    loss = update_fast(trainers, train_step)
                elif single_nn:
                    for index in range(len(trainers)):
                        trainers[0].preupdate()
                        loss = trainers[0].update(trainers, train_step, index)
                else:
                    for agent in trainers:
                        agent.preupdate()
                    for agent in trainers:
                        loss = agent.update(trainers, train_step)

            if not arglist.multicomputer_main:
                # save model, display training output
                if terminal and (len(episode_rewards) % arglist.save_rate == 0):
                    if not arglist.multicomputer_worker:
                        U.save_state(arglist.save_dir, saver=saver)
                    # print statement depends on whether or not there are adversaries
                    if num_adversaries == 0 and False:
                        print("steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
                            train_step, len(episode_rewards), np.mean(np.abs(episode_rewards[-arglist.save_rate:])), round(time.time()-t_start, 3)))
                    else:
                        print("steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(
                            train_step, len(episode_rewards), np.mean(np.abs(episode_rewards[-arglist.save_rate:])),
                            [np.mean(np.maximum(rew[-arglist.save_rate:], 0.0)) for rew in agent_rewards], round(time.time()-t_start, 3)))

                    t_start = time.time()
                    # Keep track of final episode reward
                    final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))
                    for rew in agent_rewards:
                        final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate:]))

                # saves final episode reward for plotting training curve later
                if False:
                    if len(episode_rewards) > arglist.num_episodes:
                        rew_file_name = arglist.plots_dir + arglist.exp_name + '_rewards.pkl'
                        with open(rew_file_name, 'wb') as fp:
                            pickle.dump(final_ep_rewards, fp)
                        agrew_file_name = arglist.plots_dir + arglist.exp_name + '_agrewards.pkl'
                        with open(agrew_file_name, 'wb') as fp:
                            pickle.dump(final_ep_ag_rewards, fp)
                        print('...Finished total of {} episodes.'.format(len(episode_rewards)))
                        break

if __name__ == '__main__':
    arglist = parse_args()
    train(arglist)
