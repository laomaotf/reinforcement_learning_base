# coding = utf8
import numpy as np
import gym
from matplotlib import pyplot as plt
import random
import mxnet as mx
import numpy as np

from collections import defaultdict,deque

plt.ion()

CONFIG = {

    "POOL_SIZE": 3096, #应该比较大，保存较多样本
    "BASE_LR": 0.0005, #学习率 不超过10e-3，否则容易导致每次选择同一个action
    "LAMBDA_Q1": 0.9, #表示未来可期的收益，不易太小
    "BATCH_EACH_EPOCH":128, #增加训练次数
    "PLAYING_EACH_EPOCH": 8, #每一轮进行若干轮游戏，收集样本
    "REWARD_NEG":-10, #失败时的惩罚，这个值很重要
    "LOSS":"HuberLoss",
    "TARGET_UPDATE_TAU": 0.001 #target模型更新权重
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

    def end_sampling(self):
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


class MODEL_CRITIC(mx.gluon.Block):
    def __init__(self, output_num = 2,**kwargs):
        super(MODEL_CRITIC,self).__init__(**kwargs)
        with self.name_scope():
            self.model = mx.gluon.nn.Sequential()
            self.model.add(
                mx.gluon.nn.Dense(16,prefix="custom_in",activation="relu"),
                mx.gluon.nn.Dense(32,activation="relu"),
                mx.gluon.nn.Dense(output_num, prefix="custom_out"))
        return
    def forward(self, x):
        return self.model(x)

    def copy_from(self, other, tau=CONFIG["TARGET_UPDATE_TAU"]):
        params = other.collect_params()
        params_self = self.collect_params()
        for name, param in params.items():
            key = name.replace(params.prefix, "")
            selfname = params_self.prefix + key
            #params_self[selfname].set_data(param.data())
            new_data = params_self[selfname].data() * (1 - tau) + tau * param.data()
            params_self[selfname].set_data(new_data)
        return




class MODEL_ACTOR(mx.gluon.Block):
    def __init__(self, output_num = 2,**kwargs):
        super(MODEL_ACTOR,self).__init__(**kwargs)
        with self.name_scope():
            self.model = mx.gluon.nn.Sequential()
            self.model.add(
                mx.gluon.nn.Dense(16,prefix="custom_in",activation="relu"),
                mx.gluon.nn.Dense(32,activation="relu"),
                mx.gluon.nn.Dense(output_num, prefix="custom_out"))
        return
    def forward(self, x):
        return self.model(x)

    def copy_from(self, other, tau=CONFIG["TARGET_UPDATE_TAU"]):
        params = other.collect_params()
        params_self = self.collect_params()
        for name,param in params.items():
            key = name.replace(params.prefix,"")
            selfname = params_self.prefix + key
            new_data = params_self[selfname].data() * (1 - tau) + tau * param.data()
            params_self[selfname].set_data(new_data)
        return


def PredictAction(net,S, softmax_output=False, resampling=False):
    if isinstance(S, list):
        mx_s = mx.nd.array(S)
    else:
        mx_s = mx.nd.array([S])
    probs = net(mx_s)
    if softmax_output:
        probs = mx.nd.softmax(probs,axis=-1)
    probs_np = probs.asnumpy()
    if resampling:
        action = np.random.choice(a=[0,1],size=1,p=probs_np[0])
    else:
        action = mx.nd.argmax(probs,axis=-1).asnumpy().astype(np.int32)
    probs_action = mx.nd.max(probs,axis=-1).asnumpy()
    return action,probs_action,probs_np


#Value
def TrainCritic(nets, opt, pool):
    batchs_total = CONFIG['BATCH_EACH_EPOCH']
    #batchs_total = pool.len()
    batch_size = 1
    if CONFIG["LOSS"] == "L2Loss":
        calc_q_loss = mx.gluon.loss.L2Loss()  # using regression loss!!!
    elif CONFIG["LOSS"] == "HuberLoss":
        calc_q_loss = mx.gluon.loss.HuberLoss()
    net = nets['critic']
    net_target = nets["critic_target"]
    net_actor = nets["actor_target"]
    for _ in range(batchs_total):
        batch_data = pool.get_one_batch(batch_size)
        S0 = [d[0] for d in batch_data]
        A0 = [d[1] for d in batch_data]
        R0 = [d[2] for d in batch_data]
        S1 = [d[3] for d in batch_data]
        _,_,Q1 = PredictAction(net_target,S1) #使用target_net计算Q1，明显减少训练的variance
        _, _, Q0 = PredictAction(net, S0) #Q0需要是net计算的，而不是net_target
        if 1: #OFF_POLICY
            Q1 = np.max(Q1,axis=-1)
        else:
            A1, _, _ = PredictAction(net_actor, S1)
            Q1 = Q1[:,A1]
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
    return net

#Policy
def TrainActor(nets, opt, pool):
    batchs_total = CONFIG['BATCH_EACH_EPOCH']
    #batchs_total = pool.len()
    batch_size = 1
    net_critic = nets['critic']
    net_actor = nets['actor']
    for _ in range(batchs_total):
        batch_data = pool.get_one_batch(batch_size)
        S0 = [d[0] for d in batch_data]
        A0 = [d[1] for d in batch_data]
        R0 = [d[2] for d in batch_data]
        S1 = [d[3] for d in batch_data]
        _, _, Q0 = PredictAction(net_critic, S0)
        _, _, Q1 = PredictAction(net_critic, S1)
        Q1 = np.max(Q1, axis=-1)
        if R0[0] < 0:  # batch size == 1!!
            Q1 = np.zeros_like(Q1)  # 终止条件下必须把Q

        tderr = R0 + CONFIG['LAMBDA_Q1'] * Q1 - np.max(Q0, axis=-1)

        S0_mx = mx.nd.array(S0)
        Q0_mx = mx.nd.array(tderr)
        A0_mx = mx.nd.one_hot(mx.nd.array(A0), depth=2)
        Q0_mx = A0_mx * Q0_mx
        with mx.autograd.record():
            action_probs = net_actor(S0_mx)
            action_probs = mx.nd.log_softmax(action_probs,axis=-1)
            action_probs = action_probs * Q0_mx
            loss_pg = -1 * action_probs.sum()
        loss_pg.backward()
        opt.step(batch_size)
        mx.nd.waitall()
    return net_actor



def play_one_round(env, model, sample_pool, prob_to_explore, show_wnd = False,**kwargs):
    observation = env.reset() # 重置初始状态
    sum_reward = 0 # 记录总的奖励
    observation_list = [observation]

    while True:
        if show_wnd:
            env.render()
        action,_,_ = PredictAction(model,observation_list[-1],softmax_output=True,**kwargs)
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


def play_games(epochs_total = 500):
    env = gym.make("CartPole-v0")
    np.random.seed(10)

    fig,axes = plt.subplots(2,1,figsize=(10,5))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.15, hspace=0.35)
    fig.suptitle("ActorCritic")

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
        "actor": MODEL_ACTOR(),
        "critic": MODEL_CRITIC(),
        "actor_target":MODEL_ACTOR(),
        "critic_target":MODEL_CRITIC(),
    }
    #避免mxnet的延迟初始化
    for key in models.keys():
        models[key].initialize()
        input_one = mx.nd.ones((1, 4))
        models[key](input_one)
    models['actor_target'].copy_from(models['actor'])
    models['critic_target'].copy_from(models['critic'])

    trainers = {
        "actor": mx.gluon.Trainer(models['actor'].collect_params(),"adam",{"learning_rate": CONFIG['BASE_LR'],"clip_gradient":1}),
        "critic": mx.gluon.Trainer(models['critic'].collect_params(), "adam", {"learning_rate": CONFIG['BASE_LR']})
    }


    sample_pool = SAMPLE_POOL()


    test_rewards = []
    learning_rates = []
    explore_rates = []
    for epoch in range(1,epochs_total):

        #update leanring rate
        training_progress = float(epoch) / epochs_total
        lr = CONFIG['BASE_LR'] * (1 - training_progress)
        trainers["actor"].set_learning_rate(lr)
        trainers["critic"].set_learning_rate(lr)
        learning_rates.append(lr)



        #set explore rate. exploring at first 50 epochs
        explore_rate = 1 - float(epoch) / 2.0
        explore_rate = max([0.05,explore_rate])
        explore_rates.append(explore_rate)
        #PLAY AND TRAINING
        sample_pool.reset() #AC is ON-POLICY
        for batch_num in range(CONFIG['PLAYING_EACH_EPOCH']):
            play_one_round(env, models["actor"],sample_pool, explore_rate,resampling=False)
            # TRAIN CIRTIC & ACTOR
            models['critic'] = TrainCritic(models, trainers["critic"], sample_pool)
            models['actor'] = TrainActor(models, trainers["actor"], sample_pool)
            print(sample_pool.stat_action())

        print(sample_pool.stat_action())


        # TESTING NEW MODEL
        if epoch == 1 or (0 == epoch % 10):
            rewards = 0
            for _ in range(100):
                rewards += play_one_round(env, models["actor"], None, 0, show_wnd=True,resampling=True)
            rewards /= 100.0
        else:
            rewards = test_rewards[-1]

        test_rewards.append(rewards)

        models["critic_target"].copy_from(models["critic"])
        models["actor_target"].copy_from(models["actor"])


        #DRAW TRAINING LOG
        ax.plot([k for k in range(epoch)], test_rewards,color='green',marker='*',label="reward")
        ax1.plot([k for k in range(epoch)], learning_rates, color='blue',label="leanring_rates")
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

        plt.savefig("AC.jpg")
    plt.close()
    return

play_games()






