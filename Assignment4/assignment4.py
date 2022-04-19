'''
CS7641 Assignment4: Markov Decision Process(MDP)
Author: S Yu
Date: Apr. 2022
'''


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import hiive.mdptoolbox.mdp
import hiive.mdptoolbox.example
from gym.envs.toy_text.frozen_lake import generate_random_map
import gym
import openai

def value_iter(transitions, reward, discount, epsilon, max_iter):
    vi = hiive.mdptoolbox.mdp.ValueIteration(transitions, reward, discount, epsilon, max_iter=max_iter)
    vi.run()
    stats_df = pd.DataFrame(vi.run_stats)
    # print(stats_df.tail())
    # plt.figure()
    # plt.plot(stats_df['Iteration'], stats_df['Mean V'])
    # plt.xlabel('Iterations')
    # plt.ylabel('Reward')
    # plt.title('Value Iteration')
    # plt.grid()
    # plt.tight_layout()
    # plt.savefig('Value Iteration')
    return stats_df

def policy_iter(transitions, reward, discount, max_iter):
    pi = hiive.mdptoolbox.mdp.PolicyIteration(transitions, reward, discount, max_iter=max_iter)
    pi.run()
    stats_df = pd.DataFrame(pi.run_stats)
    # print(stats_df.tail())
    # plt.figure()
    # plt.plot(stats_df['Iteration'], stats_df['Mean V'])
    # plt.xlabel('Iterations')
    # plt.ylabel('Reward')
    # plt.title('Policy Iteration')
    # plt.grid()
    # plt.tight_layout()
    # plt.savefig('Policy Iteration')
    return stats_df

def q_learning(transitions, reward, discount, epsilon = 0.3):
    ql = hiive.mdptoolbox.mdp.QLearning(transitions, reward, discount,  alpha=0.1, alpha_decay=0.99, alpha_min=0.001,
                 epsilon=epsilon, epsilon_min=0.1, epsilon_decay=0.99,skip_check=True,
                 n_iter=10000)
    ql.run()
    stats_df = pd.DataFrame(ql.run_stats)

    return stats_df

def q_learning_episode(transitions, reward, discount=0.99, epsilon = 0.95, episode = 3000):
    ql = hiive.mdptoolbox.mdp.QLearning(transitions, reward, discount,  alpha=0.5, alpha_decay=0.7, alpha_min=0.01,
                 epsilon=epsilon, epsilon_min=0.1, epsilon_decay=0.95,skip_check=True,
                 n_iter=10000)
    episode = episode
    rewards =[]
    for i in range(episode):
        ql.run()
        stats_df = pd.DataFrame(ql.run_stats)
        reward = stats_df['Mean V'].iloc[-1]
        rewards.append([i, reward])

    # print(ql.Q)
    # stats_df = pd.DataFrame(ql.run_stats)
    stats_df = pd.DataFrame(rewards, columns=['Iteration', 'Mean V'])

    print(stats_df.tail())
    return stats_df


def discount_plot(P, R, algo, problem):
    epsilon = 10**-5
    if problem == 'frozenlake':
        ITER = 10
    else:
        ITER = 100
    max_iter = 1000
    discount_lst = [0.99, 0.95, 0.9, 0.8, 0.7]
    plt.figure()
    for discount in discount_lst:
        if algo == 'Value Iteration':
            stats_df = value_iter(P, R, discount, epsilon, max_iter)
        elif algo == 'Policy Iteration':
            stats_df = policy_iter(P, R, discount, max_iter=ITER)
        elif algo == 'Qlearning':
            stats_df = q_learning(P, R, discount)
        # col_names[-2] = 'r = {}'.format(discount)
        # stats_df.columns = col_names
        plt.plot(stats_df['Iteration'],stats_df['Mean V'], label='gamma = {}'.format(discount))

    print(stats_df.columns)

    plt.legend()
    plt.grid()
    plt.xlabel('Iteration')
    plt.ylabel('Reward')
    plt.title('{}_{}'.format(problem, algo))
    plt.savefig('{}_{}_iter_discount'.format(problem,algo))



def epsilon_plot(P, R, algo,plot,problem):
    epsilon_lst = [10**-1, 10**-3, 10**-5, 10**-7, 10**-10, 10**-15,10**-20]
    max_iter = 1000
    discount = 0.95

    time_lst = []
    iter_lst = []
    plt.figure()
    for epsilon in epsilon_lst:
        if algo == 'Value Iteration':
            # stats_df = value_iter(P, R, discount, epsilon, max_iter)
            vi = hiive.mdptoolbox.mdp.ValueIteration(P, R, discount, epsilon, max_iter=max_iter)
        elif algo == 'Qlearning':
            vi = hiive.mdptoolbox.mdp.QLearning(P, R, discount, alpha=0.1, alpha_decay=0.99,
                                                alpha_min=0.001, epsilon=epsilon, epsilon_min=0.1, epsilon_decay=0.99,
                                                n_iter=10000)
        vi.run()
        # print(epsilon)
        time_lst.append(vi.time)
        iter_lst.append(vi.max_iter)
        if plot == 'reward':
            stats_df = pd.DataFrame(vi.run_stats)
            plt.plot(stats_df['Iteration'],stats_df['Mean V'], label='epsilon = {}'.format(epsilon))


        # col_names[-2] = 'r = {}'.format(discount)
        # stats_df.columns = col_names
    if plot == 'time':
        df = pd.DataFrame([epsilon_lst,time_lst,iter_lst]).T
        df.columns=['Epsilon','Time','Iteration']
        df.set_index('Epsilon', inplace=True)
        print(df.head())
        df.plot(kind='bar', secondary_y = 'Time', rot=0)
        # df.plot(kind='bar', grid=True, subplots=True, sharex=True)
        # plt.bar(epsilon_lst, time_lst, label='Time')
        # plt.hist(epsilon_lst, iter_lst, label='Iteration')



    plt.legend()
    plt.grid()
    plt.xlabel('Epsilon')
    plt.ylabel('Reward')

    plt.title('{}: Epsilon'.format(algo))
    plt.tight_layout()
    plt.savefig('{}_{}_epsilon_{}'.format(problem,algo,plot))


def epsilon_decay_plot(P, R, problem, epsilon_decay = True, episode = 1000):
    decay_lst = [0.99, 0.95, 0.9, 0.8, 0.7]
    # max_iter = 1000
    discount = 0.95
    # episode = episode
    plt.figure()
    for decay in decay_lst:
        if epsilon_decay == True:
            ql = hiive.mdptoolbox.mdp.QLearning(P, R, discount, alpha=0.1, alpha_decay=0.99,
                                                alpha_min=0.001, epsilon=0.99, epsilon_min=0.1, epsilon_decay= decay,
                                                n_iter=10000)
            label = 'epsilon_decay'
        else:
            ql = hiive.mdptoolbox.mdp.QLearning(P, R, discount, alpha=0.1, alpha_decay=0.99,
                                                alpha_min=0.001, epsilon=decay, epsilon_min=0.1, epsilon_decay=0.9,
                                                n_iter=10000)
            label = 'epsilon'

        # vi.run()
        #
        # stats_df = pd.DataFrame(vi.run_stats)
        # plt.plot(stats_df['Iteration'],stats_df['Mean V'], label='epsilon_decay = {}'.format(decay))

        rewards = []
        for i in range(episode):
            ql.run()
            stats_df = pd.DataFrame(ql.run_stats)
            reward = stats_df['Mean V'].iloc[-1]
            rewards.append([i, reward])

        # print(ql.Q)
        # stats_df = pd.DataFrame(ql.run_stats)
        stats_df = pd.DataFrame(rewards, columns=['Iteration', 'Mean V'])
        plt.plot(stats_df['Iteration'], stats_df['Mean V'], label='{} = {}'.format(label, decay))

    plt.legend()
    plt.grid()
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Qlearning: Epsilon Decay Tuning')
    plt.tight_layout()
    plt.savefig('{}_Qlearning_{}'.format(problem,label))


def qlearning_discount_plot(P, R, problem, episode = 1000):
    discount_lst = [0.99, 0.95, 0.9, 0.8, 0.7]

    # max_iter = 1000
    # discount = 0.95
    # episode = episode

    plt.figure()
    for discount in discount_lst:
        ql = hiive.mdptoolbox.mdp.QLearning(P, R, discount, alpha=0.7, alpha_decay=0.99,
                                                alpha_min=0.001, epsilon=0.99, epsilon_min=0.1, epsilon_decay= 0.9,
                                                n_iter=10000)
        # ql.run()
        #
        # stats_df = pd.DataFrame(ql.run_stats)
        # plt.plot(stats_df['Iteration'],stats_df['Mean V'], label='alpha = {}'.format(alpha))

        rewards = []
        for i in range(episode):
            ql.run()
            stats_df = pd.DataFrame(ql.run_stats)
            reward = stats_df['Mean V'].iloc[-1]
            rewards.append([i, reward])

        # print(ql.Q)
        # stats_df = pd.DataFrame(ql.run_stats)
        stats_df = pd.DataFrame(rewards, columns=['Iteration', 'Mean V'])
        plt.plot(stats_df['Iteration'],stats_df['Mean V'], label='discount = {}'.format(discount))

    plt.legend()
    plt.grid()
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Qlearning: Discount Tuning')
    plt.tight_layout()
    plt.savefig('{}_Qlearning_discount_2)'.format(problem))


def alpha_plot(P, R, problem, episode = 1000):
    alpha_lst = [0.1, 0.3, 0.5, 0.7, 0.9]
    # max_iter = 1000
    discount = 0.95
    episode = episode

    plt.figure()
    for alpha in alpha_lst:
        ql = hiive.mdptoolbox.mdp.QLearning(P, R, discount, alpha=alpha, alpha_decay=0.99,
                                                alpha_min=0.001, epsilon=0.99, epsilon_min=0.1, epsilon_decay= 0.9,
                                                n_iter=10000)
        # ql.run()
        #
        # stats_df = pd.DataFrame(ql.run_stats)
        # plt.plot(stats_df['Iteration'],stats_df['Mean V'], label='alpha = {}'.format(alpha))

        rewards = []
        for i in range(episode):
            ql.run()
            stats_df = pd.DataFrame(ql.run_stats)
            reward = stats_df['Mean V'].iloc[-1]
            rewards.append([i, reward])

        # print(ql.Q)
        # stats_df = pd.DataFrame(ql.run_stats)
        stats_df = pd.DataFrame(rewards, columns=['Iteration', 'Mean V'])
        plt.plot(stats_df['Iteration'],stats_df['Mean V'], label='alpha = {}'.format(alpha))

    plt.legend()
    plt.grid()
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Qlearning: Alpha Tuning')
    plt.tight_layout()
    plt.savefig('{}_Qlearning_alpha'.format(problem))


def alpha_decay_plot(P, R, problem, episode = 1000):
    decay_lst = [0.99, 0.95, 0.9, 0.8, 0.7]
    # max_iter = 1000
    discount = 0.95
    episode = episode
    plt.figure()
    for decay in decay_lst:
        ql = hiive.mdptoolbox.mdp.QLearning(P, R, discount, alpha=0.1, alpha_decay=decay,
                                                alpha_min=0.001, epsilon=0.9, epsilon_min=0.1, epsilon_decay= 0.99,
                                                n_iter=10000)
        # vi.run()
        #
        # stats_df = pd.DataFrame(vi.run_stats)
        # plt.plot(stats_df['Iteration'],stats_df['Mean V'], label='alpha_decay = {}'.format(decay))
        rewards = []
        for i in range(episode):
            ql.run()
            stats_df = pd.DataFrame(ql.run_stats)
            reward = stats_df['Mean V'].iloc[-1]
            rewards.append([i, reward])

        # print(ql.Q)
        # stats_df = pd.DataFrame(ql.run_stats)
        stats_df = pd.DataFrame(rewards, columns=['Iteration', 'Mean V'])
        plt.plot(stats_df['Iteration'], stats_df['Mean V'], label='alpha_decay = {}'.format(decay))

    plt.legend()
    plt.grid()
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Qlearning: Alpha Decay Tuning')
    plt.tight_layout()
    plt.savefig('{}_Qlearning_alpha_decay'.format(problem))

def error_plot(problem, algo):
    # epsilon_lst = [10 ** -1, 10 ** -3, 10 ** -5, 10 ** -7, 10 ** -10, 10 ** -15, 10 ** -20]
    epsilon_lst = [0.1, 0.2, 0.3, 0.4, 0.5]
    max_iter = 1000
    discount = 0.95
    plt.figure()
    if problem == 'forest':
        # S_lst = [100, 1000]
        # PRlst = [hiive.mdptoolbox.example.forest(S=s) for s in S_lst]
        P, R = hiive.mdptoolbox.example.forest(S=1000)

    elif problem == 'frozenlake':
        # S_lst = [64, 400]
        # env1 = openai.OpenAI_MDPToolbox('FrozenLake8x8-v1')
        # random_map = generate_random_map(size=20, p=0.98)  # map = size * size
        # env2 = openai.OpenAI_MDPToolbox('FrozenLake-v1', desc=random_map)
        # PRlst = [(env1.P, env1.R), (env2.P, env2.R)]
        env1 = openai.OpenAI_MDPToolbox('FrozenLake8x8-v1')
        P, R = env1.P, env1.R
    # for i in range(2):
    #
    #     # P, R = hiive.mdptoolbox.example.forest(S=s)
    #     P, R = PRlst[i]
    #     s = S_lst[i]
    for epsilon in epsilon_lst:
        if algo == 'Value Iteration':
            stats_df = value_iter(P, R, discount, epsilon=epsilon, max_iter=max_iter)
        elif algo == 'Policy Iteration':
            stats_df = policy_iter(P, R, discount, max_iter)
        elif algo == 'Qlearning':
            stats_df = q_learning(P, R, discount, epsilon=epsilon)
        plt.plot(stats_df['Iteration'], stats_df['Error'], label='epsilon = {}'.format(epsilon))

    # print(stats_df.columns)

    plt.legend()
    plt.grid()
    plt.xlabel('Episode')
    plt.ylabel('Error')
    plt.title('{}: Error'.format(algo))
    plt.savefig('{}_{}_iter_error'.format(problem, algo))


def state_plot(problem, algo):
    epsilon = 10 ** -5
    max_iter = 1000
    discount = 0.95

    plt.figure()
    if problem == 'forest':
        S_lst = [100, 1000]
        PRlst = [hiive.mdptoolbox.example.forest(S=s) for s in S_lst]
        ITER= 100
    elif problem == 'frozenlake':
        S_lst = [64, 400]
        env1 = openai.OpenAI_MDPToolbox('FrozenLake8x8-v1')
        random_map = generate_random_map(size=20, p=0.98) # map = size * size
        env2 = openai.OpenAI_MDPToolbox('FrozenLake-v1', desc=random_map)
        PRlst = [(env1.P, env1.R), (env2.P, env2.R)]
        ITER = 20
        print(env2.R)
    for i in range(2):

        # P, R = hiive.mdptoolbox.example.forest(S=s)
        P, R = PRlst[i]
        s=S_lst[i]
        # R[np.where(R==1)] = -1
        # R[np.where(R==0)] = -0.1

        if algo == 'Value Iteration':
            stats_df = value_iter(P, R, discount, epsilon, max_iter)
        elif algo == 'Policy Iteration':
            stats_df = policy_iter(P, R, discount, max_iter=ITER)
        elif algo == 'Qlearning':
            stats_df = q_learning(P, R, discount)
        plt.plot(stats_df['Iteration'],stats_df['Mean V'], label='state = {}'.format(s))

    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('Reward')
    plt.grid()
    plt.title('{} for different problem sizes\n gamma={}'.format(algo, discount))
    plt.savefig('{}_{}_iter_state'.format(problem,algo))


def state_plot_qlearning(problem, algo):
    epsilon = 10 ** -5
    max_iter = 1000
    discount = 0.95

    plt.figure()
    if problem == 'forest':
        S_lst = [100, 1000]
        PRlst = [hiive.mdptoolbox.example.forest(S=s) for s in S_lst]
        ITER= 100
        P, R =hiive.mdptoolbox.example.forest(S=1000)
        title = 'Forest_1000'
    elif problem == 'frozenlake':
        S_lst = [64, 400]
        env1 = openai.OpenAI_MDPToolbox('FrozenLake8x8-v1')
        random_map = generate_random_map(size=20, p=0.98) # map = size * size
        env2 = openai.OpenAI_MDPToolbox('FrozenLake-v1', desc=random_map)
        PRlst = [(env1.P, env1.R), (env2.P, env2.R)]
        ITER = 20
        # print(env2.R)
        P, R = env1.P, env1.R
        title = 'FrozenLake8x8'


    if algo == 'Value Iteration':
        stats_df = value_iter(P, R, discount, epsilon, max_iter)
    elif algo == 'Policy Iteration':
        stats_df = policy_iter(P, R, discount, max_iter=ITER)
    elif algo == 'Qlearning':
        stats_df = q_learning_episode(P, R, discount)
    plt.plot(stats_df['Iteration'],stats_df['Mean V'])

    # plt.legend()
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid()
    plt.title('{}:{} Episode'.format(title,algo))
    plt.savefig('{}_{}_episode'.format(problem,algo))


def time_plot(problem, algo):
    epsilon = 10 ** -5
    max_iter = 1000
    discount = 0.95
    plt.figure()

    if problem == 'forest':
        S_lst = [100, 1000]
        PRlst = [hiive.mdptoolbox.example.forest(S=s) for s in S_lst]
        ITER = 100

    elif problem == 'frozenlake':
        S_lst = [64, 400]
        env1 = openai.OpenAI_MDPToolbox('FrozenLake8x8-v1')
        random_map = generate_random_map(size=20, p=0.98) # map = size * size
        env2 = openai.OpenAI_MDPToolbox('FrozenLake-v1', desc=random_map)
        PRlst = [(env1.P, env1.R), (env2.P, env2.R)]
        ITER = 20

    for i in range(2):
        # P, R = hiive.mdptoolbox.example.forest(S=s)
        P, R = PRlst[i]
        s=S_lst[i]

        if algo == 'Value Iteration':
            stats_df = value_iter(P, R, discount, epsilon, max_iter)
        elif algo == 'Policy Iteration':
            stats_df = policy_iter(P, R, discount, max_iter=ITER)
        elif algo == 'Qlearning':
            stats_df = q_learning(P, R, discount)

        plt.plot(stats_df['Iteration'],stats_df['Time'], label='state = {}'.format(s))

    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('Time')
    plt.grid()
    plt.title('Time of {}'.format(algo))
    plt.savefig('{}_{}_iter_time'.format(problem, algo))

if __name__ == '__main__':
    t1 = time.time()
    np.random.seed(5)

    pd.set_option('max_columns', None)

    problems = ['forest', 'frozenlake']
    algorithms = ['Value Iteration', 'Policy Iteration', 'Qlearning']

    for problem in problems:
        if problem == 'forest':
            ## Non-gridworld problem - forest
            P, R = hiive.mdptoolbox.example.forest(S=1000)
        else:

            env = openai.OpenAI_MDPToolbox('FrozenLake8x8-v1')
            # random_map = generate_random_map(size=20, p=0.98) # map = size * size
            # env = openai.OpenAI_MDPToolbox('FrozenLake-v1', desc=random_map)
            P = env.P
            R = env.R
        # R[np.where(R==1)] = -0.1
        # print(R)
        # for algo in algorithms:
            # discount_plot(P, R, algo, problem)
            # state_plot(problem, algo)
            # time_plot(problem, algo)
            # error_plot(problem, algo)

        # # # epsilon is applicable for VI and QL
        # epsilon_plot(P, R, 'Value Iteration','time',problem)
        # epsilon_plot(P, R, 'Value Iteration','reward',problem)
        #
        # epsilon_plot(P, R, 'Qlearning', 'reward',problem)
        # epsilon_plot(P, R, 'Qlearning','time',problem)
        #
        # # only applicable for QL
        # qlearning_discount_plot(P, R,  problem,episode = 100)
        # epsilon_decay_plot(P, R,problem, epsilon_decay= True, episode = 100)
        # epsilon_decay_plot(P, R,problem, epsilon_decay= False, episode = 100)
        # alpha_plot(P, R,problem)
        # alpha_decay_plot(P, R,problem,episode = 100)

    # P, R = hiive.mdptoolbox.example.forest(S=1000)
    # discount_plot(P, R, 'Policy Iteration', 'forest')

    # discount_plot(P, R, 'Value Iteration')
    # discount_plot(P, R, 'Policy Iteration')
    # discount_plot(P, R, 'Qlearning')
    #
    # time_plot('frozenlake', 'Policy Iteration')
    # time_plot('forest', 'Policy Iteration')

    state_plot_qlearning('forest', 'Qlearning')
    # state_plot_qlearning('frozenlake', 'Qlearning')
    #
    # state_plot('frozenlake', 'Value Iteration')
    # state_plot('frozenlake', 'Policy Iteration')

    # state_plot('frozenlake', 'Qlearning')
    # state_plot('forest', 'Qlearning')
    #
    # #
    # time_plot(S_lst, 'Value Iteration')
    # time_plot(S_lst, 'Policy Iteration')
    # time_plot(S_lst, 'Qlearning')
    #
    # error_plot(S_lst, 'Value Iteration')
    # error_plot(S_lst, 'Policy Iteration')
    # error_plot(S_lst, 'Qlearning')

    # discount=0.95
    # q_learning(P, R, discount)

    t2 = time.time()
    print('Run Time: {}'.format(t2-t1))