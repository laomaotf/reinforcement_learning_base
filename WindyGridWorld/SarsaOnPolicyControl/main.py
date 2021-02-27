# encoding=utf-8
import os,sys
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import Counter

#交互式显示
plt.ion()
plt.figure("SARSA")

WORLD_WIDTH, WORLD_HEIGHT = 10, 7
START_POS, STOP_POS = (0,3),(7,3)
TARGET_REWARD = 100


class ENV(object):
    def __init__(self):
        self.wind = np.zeros((WORLD_HEIGHT,WORLD_WIDTH,2),dtype=np.float32)
        self.S = START_POS #x,y
        self.E = STOP_POS
        self.wind[:,3:6,:] = (0,-1)
        self.wind[:,8,:] = (0,-1)
        self.wind[:,6:8,:] = (0, -2)
        self.reset()
        return
    def reset(self):

        return
    def update(self,xy):
        #输入当前状态
        #返回收益,新的状态,结束标志
        x0,y0 = [int(a) for a in xy]
        x0 = np.clip(x0,0, WORLD_WIDTH-1)
        y0 = np.clip(y0,0, WORLD_HEIGHT-1)
        if x0 == self.E[0] and y0 == self.E[1]:
            R = TARGET_REWARD #game over, agent success
            ending = True
        else:
            R = -1
            ending = False

        if ending:
            return R, (x0,y0), ending

        x1,y1 = (x0 + self.wind[y0,x0,0], y0 + self.wind[y0,x0,1])
        if x1 == self.E[0] and y1 == self.E[1]:
            R = TARGET_REWARD  # game over, agent success
            ending = True

        x1 = np.clip(x1,0, WORLD_WIDTH-1)
        y1 = np.clip(y1,0, WORLD_HEIGHT-1)

        return R, (x1, y1), ending
import warnings
class AGENT(object):
    def __init__(self):
        self.min_eps = 0.05
        self.act_num = 4
        self.Q = np.zeros((WORLD_HEIGHT,WORLD_WIDTH,self.act_num),dtype=np.float32) #UP,BOTTOM,LEFT,RIGHT
        return
    def reset(self):
        return
    def _convert_code_to_action(self,code):
        if code == 0:
            return (0,-1,0) #dx,dy,code
        elif code == 1:
            return (0,1,1)
        elif code == 2:
            return (-1,0,2)
        elif code == 3:
            return (1,0,3)
        warnings.warn(f">>UNK action code: {code}")
        return (0,0,code)
    def take_action(self,xy,greed_eps=1.0):

        if greed_eps < self.min_eps:
            greed_eps  = self.min_eps

        x,y = [int(a) for a in xy]

        #in-place
        action = np.argmax(self.Q[y,x,:])
        if random.uniform(0, 1) < greed_eps:
            action = random.randint(0, self.act_num-1) #eps greedy
        return self._convert_code_to_action(action)

import copy
class TRAIN_Q_SARSA(object):
    def __init__(self, discount = 0.9,lr = 0.5):
        self.discount = discount
        self.lr = lr
        return
    def train_agent(self,agent, trajs):
        Q_old = copy.deepcopy(agent.Q)
        Q = agent.Q
        for traj in trajs:
            for (s0,a0,r,s1,a1) in traj:
                x1,y1 = [int(dd) for dd in s1]
                a1 = int(a1[-1])
                if a1 >= 0:
                    q1 = Q_old[y1,x1,a1]
                else:
                    q1 = 0
                x0,y0 = [int(dd) for dd in s0]
                a0 = int(a0[-1])
                Q[y0,x0,a0] = Q[y0,x0,a0] + self.lr * (r + self.discount * q1 - Q[y0,x0,a0])
        agent.Q= Q
        return









class GAME(object):
    def __init__(self):
        self.samples = []
        self.env = ENV()
        self.agent = AGENT()
        return
    def reset(self):
        self.env.reset()
        return
    def play(self,**kwargs):
        traj = []
        S0 = self.env.S
        while True:
            A0 = self.agent.take_action(S0,**kwargs) #action 选择action
            R0, S1, ending = self.env.update( (S0[0] + A0[0],S0[1] + A0[1])) #环境对新状态返回reward，以及下一个状态
            A1 = self.agent.take_action(S1,greed_eps=0.0) #按照当前规则选择下一个动作
            if ending or len(traj) > WORLD_WIDTH * WORLD_HEIGHT*1000:
                traj.append((S0,A0,R0,S1,A1))
                return traj
            traj.append((S0,A0,R0, S1,A1))
            S0 = S1
        return traj

def show_traj(traj):
    line = []
    for (s0,a0, r0, s1,a1) in traj[0:5]:
        line.append("(sa:{} a:{} r:{} sb:{} ab:{})".format(s0,a0,r0,s1,a1))
        #break
    print(','.join(line))

    X, Y = [],[]
    for (s0,a0, r0, s1,_) in traj:
        X.append(s0[0])
        Y.append(s0[1])
        if r0 == TARGET_REWARD:
            X.append(s1[0])
            Y.append(s1[1])
    plt.cla()
    plt.title(f'traj length {len(traj)}')
    plt.scatter(START_POS[0], START_POS[1],marker="s",color="blue",s = 150)
    plt.scatter(STOP_POS[0], STOP_POS[1], marker="o", color="red", s=150)
    plt.plot(X,Y,color='green')
    plt.grid()
    plt.ylim((-1,WORLD_HEIGHT+1))
    plt.xlim((-1,WORLD_WIDTH+1))
    plt.xticks(range(-1,WORLD_WIDTH+1))
    plt.pause(1)
    return line


def train_agent(loops_total = 100, epsiodes_each_loop = 10):
    game = GAME()
    trainer = TRAIN_Q_SARSA()

    for loop in range(loops_total):
        traj_all = []
        win_count = 0
        greed_eps = 1.0 - loop*5 / float(loops_total) #利用模拟退化处理greedy算法，初始阶段依赖随机算法，后期依赖训练结果
        for _ in range(epsiodes_each_loop):
            game.reset()
            traj_one = game.play(greed_eps=greed_eps)
            if traj_one[-1][2] > 0:
                win_count += 1
            #show_traj(traj_one)
            traj_all.append(traj_one)
        print(f"win {win_count} greed eps {greed_eps}")
        trainer.train_agent(game.agent,traj_all)


        if (loop % 1) == 0:
            game.reset()
            traj = game.play(greed_eps=0)
            show_traj(traj)


if __name__=="__main__":
    train_agent()



