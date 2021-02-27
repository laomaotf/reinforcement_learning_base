# CartPole-v0
[游戏规则](https://gym.openai.com/envs/CartPole-v0/)  

A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The system is controlled by applying a force of +1 or -1 to the cart. The pendulum starts upright, and the goal is to prevent it from falling over. A reward of +1 is provided for every timestep that the pole remains upright. The episode ends when the pole is more than 15 degrees from vertical, or the cart moves more than 2.4 units from the center.

## 状态
(x, $\theta$ , $\bar x$ $\bar \theta$)   
依次表示
(小车未知，杆子和竖直方向夹角，小车速度，角度变化率)   

## 动作
动作只有两个：向左，向右
 
## 激励
每一步都会获得激励1，而且返回的done=True时，激励也是1

## GAME OVER条件
* reward累积200
* 小车位置偏移中心2.4个单位：移动出窗口
* 小车和竖直方向夹角大于2.4：杆子倾倒

## 获胜标准
连续100轮游戏的平均收益不低于195.0