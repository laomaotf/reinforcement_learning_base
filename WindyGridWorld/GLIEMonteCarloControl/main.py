# encoding=utf-8
import os,sys
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import Counter
import warnings

#交互式显示
plt.ion()
plt.figure("Agent's Q(s,a)")

WORLD_WIDTH, WORLD_HEIGHT = 10, 7
START_POS, STOP_POS = (0,3),(7,3)
TARGET_REWARD = 100

AGENT_UPDATE_MODE = "not-full_backup"

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

class AGENT(object):
    def __init__(self):
        self.min_eps = 0.05
        self.act_num = 4
        self.Q = np.zeros((WORLD_HEIGHT,WORLD_WIDTH,self.act_num),dtype=np.float32) #UP,BOTTOM,LEFT,RIGHT
        self.N = np.zeros((WORLD_HEIGHT,WORLD_WIDTH,self.act_num),dtype=np.float32)
        self.Q_full = np.zeros((WORLD_HEIGHT,WORLD_WIDTH,self.act_num),dtype=np.float32)
        self.mode = AGENT_UPDATE_MODE
        return
    def reset(self):
        # self.Q = np.zeros((WORLD_HEIGHT,WORLD_WIDTH,self.act_num),dtype=np.float32) #UP,BOTTOM,LEFT,RIGHT
        # self.N = np.zeros((WORLD_HEIGHT,WORLD_WIDTH,self.act_num),dtype=np.float32)
        # self.Q_full = np.zeros((WORLD_HEIGHT,WORLD_WIDTH,self.act_num),dtype=np.float32)
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
    def take_action(self,xy,greed_eps=1.0,train_mode=True):

        if greed_eps < self.min_eps:
            greed_eps  = self.min_eps

        x,y = [int(a) for a in xy]

        if self.mode == "full_backup":
            valid = self.N > 0
            Q = np.zeros((WORLD_HEIGHT,WORLD_WIDTH,self.act_num),dtype=np.float32) - 10000
            Q[valid] = self.Q_full[valid] / self.N[valid]
            action = np.argmax(Q[y,x,:])
            if train_mode and random.uniform(0,1) < greed_eps:
                action = random.randint(0, self.act_num-1)  #eps greedy
            return self._convert_code_to_action(action)
        #in-place
        action = np.argmax(self.Q[y,x,:])
        if train_mode and random.uniform(0, 1) < greed_eps:
            action = random.randint(0, self.act_num-1) #eps greedy
        return self._convert_code_to_action(action)


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
                x,y = [int(dd) for dd in s[0:2]]
                a = int(a[-1])
                n = N[y,x,a]
                q = Q[y,x,a]
                Q_full[y,x,a] += g
                Q[y,x,a] = (q * n + g)/float(n+1)  #in-place update
                N[y,x,a] = n + 1
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
        #self.agent.reset()
        return
    def play(self,**kwargs):
        traj = []
        S0 = self.env.S
        while True:
            A0 = self.agent.take_action(S0,**kwargs) #action 选择action
            R0, S1, ending = self.env.update( (S0[0] + A0[0],S0[1] + A0[1])) #环境对新状态返回reward，以及下一个状态
            if ending or len(traj) > WORLD_WIDTH * WORLD_HEIGHT*1000:
                traj.append((S0,A0,R0,S1))
                return traj
            traj.append((S0,A0,R0, S1))
            S0 = S1
        return traj

def show_traj(traj):
    line = []
    for (s0,a0, r0, s1) in traj[0:5]:
        line.append("(sa:{} a:{} r:{} sb:{})".format(s0,a0,r0,s1))
        #break
    print(','.join(line))

    X, Y = [],[]
    for (s0,a0, r0, s1) in traj:
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

# def show_agent(agent):
#     plt.cla()
#     Q = agent.Q
#     S,A = Q.shape
#     X,Y0,Y1 = [],[],[]
#     for s in range(S):
#         X.append(s)
#         a0,a1 = Q[s,0] - Q[s,:].min(), Q[s,1] - Q[s,:].min()
#         Y0.append(a0 / (a1 + a0))
#         Y1.append(a1 / (a1 + a0))
#     barw = 0.4
#     plt.bar([x - 0 for x in X], Y0, barw, align="center",label="stay")
#     plt.bar([x + barw for x in X], Y1, barw, align="center", label="fwd")
#     plt.xticks(X)
#     plt.grid()
#     plt.ylim((-1.1,1.1))
#     plt.legend()
#     plt.pause(1)



def train_agent(loops_total = 100, epsiodes_each_loop = 10):
    game = GAME()
    trainer = TRAIN_Q_MC()

    for loop in range(loops_total):
        traj_all = []
        win_count = 0
        greed_eps = 1.0 - loop / float(loops_total) #利用模拟退化处理greedy算法，初始阶段依赖随机算法，后期依赖训练结果
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
            traj = game.play(train_mode=False,greed_eps=0)
            show_traj(traj)



if __name__=="__main__":
    train_agent()



