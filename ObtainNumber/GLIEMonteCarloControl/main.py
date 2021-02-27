# encoding=utf-8
import os,sys
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import Counter

#交互式显示
plt.ion()
plt.figure("Agent's Q(s,a)")

THE_NUMBER = 21

ENV_POLICY = "not_random"
AGENT_UPDATE_MODE = "not_full_backup" #玩1K次旧可以收敛

class ENV(object):
    def __init__(self):
        self.policy = ENV_POLICY
        return
    def reset(self):
        return
    def update_rand(self,current):
        if random.uniform(0,1) > 0.5:
            return current
        return current + 1
    def update_opt(self,current):
        if current % 3 == 0:
            return current
        elif (current+1)%3 == 0:
            return current + 1
        return np.random.randint(current,current + 1 + 1)
    def update(self,current):
        #输入当前状态
        #返回收益,新的状态,结束标志

        if current >= THE_NUMBER:
            R = 1 #game over, agent success
            ending = True
        elif current >= THE_NUMBER - 2:
            R = -1 #游戏特性，在对手理性条件下，对手是必定获胜
            ending = True
        else:
            R = 0
            ending = False

        if ending:
            return R, -1, ending

        S = current + 1 #下一个状态的取值范围[current+1, current+2]
        if self.policy == "random":
            S = self.update_rand(S)
        else:
            S = self.update_opt(S)

        if S >= THE_NUMBER:
            R = -1 #game over, agent failure
            ending = True
            return R, S, ending

        return R,S+1,ending

class AGENT(object):
    def __init__(self,eps = 0.1):
        self.Q = np.zeros((THE_NUMBER+1,2))
        self.N = np.zeros((THE_NUMBER+1,2))
        self.Q_full = np.zeros((THE_NUMBER+1,2))
        self.eps = eps
        self.mode = AGENT_UPDATE_MODE
        return
    def reset(self):
        self.Q = np.zeros((THE_NUMBER+1,2))
        self.N = np.zeros((THE_NUMBER+1, 2))
        self.Q_full = np.zeros((THE_NUMBER + 1, 2))
        return
    def take_action(self,current,train_mode=True):

        if self.mode == "full_backup":
            valid = self.N > 0
            Q = np.zeros((THE_NUMBER+1,2)) - 100
            Q[valid] = self.Q_full[valid] / self.N[valid]
            action = np.argmax(Q[current,:])
            if train_mode and random.uniform(0,1) < self.eps:
                return 1 - action #eps-greedy
            return action
        #in-place
        action = np.argmax(self.Q[current,:])
        if train_mode and random.uniform(0,1) < self.eps:
            return 1 - action #eps-greedy
        return action


class TRAIN_Q_MC(object):
    def __init__(self, discount = 0.9):
        self.discount = discount
        return
    def _calc_traj_G(self, traj):
        '''
        G(i) = E[r(i+1) + discount*r(i+2) + discount^2*r(i+3) ]
        '''
        traj_g = []
        G = None
        for (s0, a0, r0, s1) in traj[::-1]:
            if G is None:
                G = r0
                traj_g.append((s0,a0,G))
            else:
                G = r0 + G * self.discount
                traj_g.append((s0,a0,G))
        return traj_g

    def train_agent(self,agent, trajs):
        Q,N,Q_full = agent.Q, agent.N, agent.Q_full
        for traj in trajs:
            traj_g = self._calc_traj_G(traj)
            for (s,a, g) in traj_g:
                n = N[s,a]
                q = Q[s,a]
                Q_full[s,a] += g
                Q[s,a] = (q * n + g)/float(n+1)  #in-place update
                N[s, a] = n + 1
        agent.Q= Q
        agent.N = N
        agent.Q_full = Q_full









class GAME(object):
    def __init__(self):
        self.samples = []
        self.env = ENV()
        self.agent = AGENT()
        return
    def reset(self):
        self.env.reset()
        return
    def play(self,start_state,**kwargs):
        traj = []
        S0 = start_state
        while S0 >= 0:
            A0 = self.agent.take_action(S0,**kwargs) #action 选择action
            R0, S1, ending = self.env.update(S0 + A0) #环境对新状态返回reward，以及下一个状态
            if ending:
                traj.append((S0,A0,R0,-1))
                return traj
            traj.append((S0,A0,R0, S1))
            S0 = S1
        return traj

def show_traj(traj):
    line = []
    for (s0,a0, r0, s1) in traj:
        line.append("(sa:{} a:{} r:{} sb:{})".format(s0,a0,r0,s1))
    print(','.join(line))
    return line

def show_agent(agent):
    plt.cla()
    Q = agent.Q
    S,A = Q.shape
    X,Y0,Y1 = [],[],[]
    for s in range(S):
        X.append(s)
        a0,a1 = Q[s,0] - Q[s,:].min(), Q[s,1] - Q[s,:].min()
        Y0.append(a0 / (a1 + a0))
        Y1.append(a1 / (a1 + a0))
    barw = 0.4
    plt.bar([x - 0 for x in X], Y0, barw, align="center",label="stay")
    plt.bar([x + barw for x in X], Y1, barw, align="center", label="fwd")
    plt.xticks(X)
    plt.grid()
    plt.ylim((-1.1,1.1))
    plt.legend()
    plt.pause(1)



def train_agent(loops_total = 1000, epsiodes_each_loop = 10):
    #agent = AGENT()
    game = GAME()
    trainer = TRAIN_Q_MC()

    for loop in range(loops_total):
        traj_all = []
        for _ in range(epsiodes_each_loop):
            game.reset()
            start_state = random.randint(1,THE_NUMBER//4)
            traj_one = game.play(start_state)
            #show_traj(traj_one)
            traj_all.append(traj_one)
        trainer.train_agent(game.agent,traj_all)


        if (loop % 10) == 0:
            show_agent(game.agent)
            game_results = []
            for _ in range(10):
                game.reset()
                #start_state = random.randint(1, THE_NUMBER//4)
                start_state = 0 #只有从0开始 player才有机会获胜，否则在最优策略ENV一定会获胜
                traj = game.play(start_state,train_mode=False)
                _,_,r,_ = traj[-1]
                if r > 0:
                    game_results.append(1)
                else:
                    game_results.append(0)

            game_results = Counter(game_results)
            line = ">>{} TEST, WIN {}, LOSE {}".format(loop*epsiodes_each_loop, game_results[1], game_results[0])
            print(line)



if __name__=="__main__":
    train_agent()



