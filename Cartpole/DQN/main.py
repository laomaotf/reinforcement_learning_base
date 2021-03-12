# coding = utf8
import numpy as np
import gym
import time
from matplotlib import pyplot as plt
import random
import mxnet as mx
import numpy as np
import json
import os
from collections import defaultdict,deque

plt.ion()

CONFIG = {

    "POOL_SIZE": 1024, #应该比较大，保存较多样本
    "BASE_LR": 0.001, #学习率 10e-3左右,不易过大
    "LAMBDA_Q1": 0.9, #表示未来可期的收益，不易太小
    "COPY_EVERY_TRAINING": 1, #训练n次后，复制一次网络
    "TEST_EVERY_TRAINING": 10, #训练n次后，测试一次网络
    "BATCH_EACH_EPOCH_MAX":256, #不易超过steps in one epochs
    "PLAYING_EACH_EPOCH": 8, #每一轮进行若干轮游戏，收集样本
    "REWARD_NEG":-10, #失败时的惩罚，这个值很重要
    "BALANCED_SAMPLING":False,
    "LOSS":"HuberLoss",
    "TARGET_UPDATE_TAU": 0.2 #target模型更新权重，减小可以提高训练稳定性，加速收敛
}

class SAMPLE_POOL:
    def __init__(self):
        self.samples = deque(maxlen=CONFIG['POOL_SIZE'])
        return
    def reset(self):
        self.samples = []
    def add(self,x):
        #s0,a0,r,s1 = x
        self.samples.append(x)
        return
    def begin_sampling(self):
        return
        #
        # self.samples_raw = []
        # self.samples_raw.extend(self.samples)
        #
        # if CONFIG["BALANCED_SAMPLING"]:
        #     self.samples = []
        #     splits = defaultdict(list)
        #     for one in self.samples_raw:
        #         action = one[1]
        #         splits[action].append(one)
        #     N = len(self.samples_raw) // len(splits.keys())
        #     for action in splits.keys():
        #         selected = np.random.randint(0,len(splits[action]),N)
        #         for ind in selected:
        #             self.samples.append(splits[action][ind])
        #
        # return
    def end_sampling(self):
        # self.samples = []
        # self.samples.extend(self.samples_raw)
        return
    def stat_action(self):
        actions = defaultdict(int)
        for (s0,a0,r,s1) in self.samples:
            actions[a0] += 1
        return actions
    def get_one_batch(self,batch_size, replace=True):
        N = len(self.samples)
        indexes = [k for k in range(N)]
        indexes = np.random.choice(indexes,batch_size,replace=replace)
        return [self.samples[k] for k in indexes]

    def len(self):
        return len(self.samples)


class MODEL(mx.gluon.Block):
    def __init__(self, output_num = 2,**kwargs):
        super(MODEL,self).__init__(**kwargs)
        with self.name_scope():
            self.model = mx.gluon.nn.Sequential()
            self.model.add(
                mx.gluon.nn.Dense(16,prefix="custom_in",activation="relu"),
                mx.gluon.nn.Dense(32,activation="relu"),
                mx.gluon.nn.Dense(output_num, prefix="custom_out"))
        return
    def forward(self, x):
        return self.model(x)

    def copy_from(self, other):
        tau = CONFIG["TARGET_UPDATE_TAU"]
        params = other.collect_params()
        params_self = self.collect_params()
        for name,param in params.items():
            key = name.replace(params.prefix,"")
            selfname = params_self.prefix + key
            #params_self[selfname].set_data(param.data())
            new_data = params_self[selfname].data() * (1 - tau) + tau * param.data()
            params_self[selfname].set_data(new_data)
        return


def PredictAction(net,S):
    if isinstance(S, list):
        mx_s = mx.nd.array(S)
    else:
        mx_s = mx.nd.array([S])
    probs = net(mx_s)
    probs_np = probs.asnumpy()
    action = mx.nd.argmax(probs,axis=-1).asnumpy().astype(np.int32)
    probs_action = mx.nd.max(probs,axis=-1).asnumpy()
    return action,probs_action,probs_np



def Train_QLearning(nets, opt, pool):
    batchs_total = min([CONFIG['BATCH_EACH_EPOCH_MAX'],pool.len()])
    batch_size = 1
    if CONFIG["LOSS"] == "L2Loss":
        calc_q_loss = mx.gluon.loss.L2Loss()  # using regression loss!!!
    elif CONFIG["LOSS"] == "HuberLoss":
        calc_q_loss = mx.gluon.loss.HuberLoss()
    net = nets["qnet"]
    net_target = nets["target"]
    for _ in range(batchs_total):
        batch_data = pool.get_one_batch(batch_size)
        S0 = [d[0] for d in batch_data]
        A0 = [d[1] for d in batch_data]
        R0 = [d[2] for d in batch_data]
        S1 = [d[3] for d in batch_data]
        _,_,Q1 = PredictAction(net_target,S1)
        _, _, Q0 = PredictAction(net, S0) #Q0需要是net计算的，而不是net_target
        Q1 = np.max(Q1,axis=-1)
        if R0[0] < 0: #batch size == 1!!
            Q1 = np.zeros_like(Q1) # 终止条件下必须把Q1设置成0，否则最开始还ok，但是随着网络输出越来越大，Q1的影响会超过Reward

        Q0[:,A0] =  R0 + CONFIG['LAMBDA_Q1'] * Q1
        S0_mx = mx.nd.array(S0)
        Q0_mx = mx.nd.array(Q0)
        with mx.autograd.record():
            Q0bar = net(S0_mx)
            loss_q = calc_q_loss(Q0bar,Q0_mx)
        loss_q.backward()
        opt.step(batch_size)
        mx.nd.waitall()
        #print("{}".format(loss_q.sum().asscalar()))
    return net






def play_one_round(env, model, sample_pool, prob_to_explore, show_wnd = False):
    observation = env.reset() # 重置初始状态
    sum_reward = 0 # 记录总的奖励
    observation_list = [observation]

    while True:
        if show_wnd:
            env.render()
        action,_,_ = PredictAction(model,observation_list[-1])
        action = action[0]
        rnd = random.uniform(0,1)
        if rnd < prob_to_explore:
            action = random.choice([0,1]) # explore more options

        observation, reward, done, info = env.step(action)

        observation_list.append(observation)

        if  not (sample_pool is None):
            if not done:
                #S0,A0,R,S1
                sample_pool.add((observation_list[-2],action,reward,observation_list[-1]))
            else:
                sample_pool.add((observation_list[-2],action,CONFIG['REWARD_NEG'], observation_list[-1])) #导致游戏失败，S1=None

        sum_reward += reward

        if done:# game over
            break


    return sum_reward


def play_games(epochs_total = 1000):
    env = gym.make("CartPole-v0")
    np.random.seed(10)

    fig,axes = plt.subplots(2,1,figsize=(10,5))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.15, hspace=0.35)
    fig.suptitle("DQN")

    ax = axes[0]
    ax.set_title("reward VS learning-rate")
    ax.set_ylim([0, 200])
    ax.set_xlim([0,50])
    ax.grid()


    ax1 = ax.twinx()
    ax1.set_ylim([0, CONFIG["BASE_LR"]])
    ax1.set_xlim([0,50])



    bx = axes[1]
    bx.set_title("reward VS exploring-rate")
    bx.set_ylim([0, 200])
    bx.set_xlim([0,50])
    bx.grid()


    bx1 = bx.twinx()
    bx1.set_ylim([0, 1])
    bx1.set_xlim([0,50])
    #ax1.set_ylabel("lr")
    #bx1.set_xlabel("epoch")


    models = {
        "qnet": MODEL(),
        "target": MODEL()
    }
    #避免mxnet的延迟初始化
    for key in models.keys():
        models[key].initialize()
        input_one = mx.nd.ones((1, 4))
        models[key](input_one)
    models["target"].copy_from(models["qnet"])

    trainer = mx.gluon.Trainer(models["qnet"].collect_params(),
                               "adam",
                               {"learning_rate": CONFIG['BASE_LR']})

    sample_pool = SAMPLE_POOL()


    test_rewards = []
    leanring_rates = []
    explore_rates = []
    for epoch in range(1,epochs_total):

        #update leanring rate
        training_progress = float(epoch) / epochs_total
        lr = CONFIG['BASE_LR'] * (1 - training_progress)
        trainer.set_learning_rate(lr)
        leanring_rates.append(lr)


        #set explore rate. exploring at first 50 epochs
        explore_rate = 1 - float(epoch) / 50.0
        explore_rate = max([0.05,explore_rate])
        explore_rates.append(explore_rate)
        #PLAY AND TRAINING
        #sample_pool.reset()
        for batch_num in range(CONFIG['PLAYING_EACH_EPOCH']):
            play_one_round(env, models["qnet"],sample_pool, explore_rate)
            sample_pool.begin_sampling()
            models["qnet"] = Train_QLearning(models, trainer, sample_pool)
            sample_pool.end_sampling()
            #print(sample_pool.stat_action())

        # UPDATE TARGET MODEL
        models["target"].copy_from(models["qnet"])

        if test_rewards == [] or epoch % CONFIG['TEST_EVERY_TRAINING'] == 0:
            #TESTING NEW MODEL
            rewards = 0
            for _ in range(100):
                rewards += play_one_round(env, models["target"], None,0,show_wnd=True)
            test_rewards.append(rewards /100.0)
        else:
            test_rewards.append(test_rewards[-1]) #copy reward from last epoch if not testing



        #DRAW TRAINING LOG
        ax.plot([k for k in range(epoch)], test_rewards,color='green',marker='*',label="reward")
        ax1.plot([k for k in range(epoch)], leanring_rates, color='blue',label="leanring_rates")
        maxx = (epoch + 49) // 50 * 50
        ax.set_xlim([0,maxx])
        ax1.set_xlim([0,maxx])


        bx.plot([k for k in range(epoch)], test_rewards,color='green',marker='*',label="reward")
        bx1.plot([k for k in range(epoch)], explore_rates, color='red', label="exploring rate")
        #maxx = (epoch + 49) // 50 * 50
        bx.set_xlim([0,maxx])
        bx1.set_xlim([0,maxx])

        if epoch == 1:
            ax1.legend(loc='center right')
            ax.legend(loc='center left')

            bx1.legend(loc='center right')
            bx.legend(loc='center left')

        plt.savefig("DQN_TRAIN.jpg")





    plt.close()
    return

play_games()






