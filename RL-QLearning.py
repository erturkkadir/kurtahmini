import numpy as np
import pandas as pd
import time

np.random.seed(2)


N_STATES = 6
ACTIONS = ['sol', 'sag']

EPSILON = 0.9
ALPHA = 0.1 # ogrenme orani
GAMMA = 0.9 # discount factor

MAX_EPISODES = 15
FRESH_TIME = 0.3


def build_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))), columns=actions)
    return table


def choose_action(state, q_table):
    state_actions = q_table.iloc[state, :]
    # print(state_actions)
    if (np.random.uniform() > EPSILON) or ( state_actions==0).all():
        action_name = np.random.choice(ACTIONS)
    else:
        action_name = state_actions.idxmax()
    print(" Action name : ", action_name)
    return action_name


def get_env_feedback(S, A):
    if A=='sag':
        if S == N_STATES-2:
            S_ = 'terminal'
            R = 1
        else:
            S_ = S + 1
            R = 0
    else:
        R = 0
        if S == 0:
            S_ = S
        else:
            S_ =  S - 1
    return S_, R


def update_env(S, episode, step_counter):
    env_list = ['X']*(N_STATES-1)+['T']
    if S == 'terminal':
        interaction = 'Episode is%s tatal_steps = %s' % (episode+1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                            ', end='')
    else:
        env_list[S] = 'O'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)


def rl():
    q_table = build_q_table(N_STATES, ACTIONS)
    for episode in range(MAX_EPISODES):
        step_counter = 0
        S = 0
        is_terminal = False
        update_env(S, episode, step_counter)
        while not is_terminal:
            A = choose_action(S, q_table)
            S_, R = get_env_feedback(S, A)
            q_predict = q_table.loc[S, A]
            if S_ != 'terminal':
                q_target = R + GAMMA * q_table.iloc[S_, :].max()
            else:
                q_target = R
                is_terminal = True
            q_table.loc[S, A] += ALPHA * (q_target - q_predict)
            S = S_
            update_env(S, episode, step_counter+1)
    return q_table


if __name__ == "__main__":
    q_table = rl()
    print('\r\nQ_Table:\n')
    print(q_table)