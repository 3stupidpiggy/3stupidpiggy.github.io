# RL算法

## Q-learning

折扣率γ  学习率α

当前状态s,执行动作a，Q值Q1 ——> 下一状态s‘

取下一状态s’所有动作中最大的Q值Q2

TD差：TD = reward+ γ*Q2 -Q1

更新Q1：Q1’ = Q1+ α*TD

### 贪心策略

给定一个概率a，用1-a的概率选择当前价值最高的动作，以a的概率选择随机动作

意义：可以发现更好的动作

实现：

```python
 
if  np.random.uniform(0,1) < 1-a :
	action = np.argmax(self.q_table[state][:])
else:
	action = np.ranom.choice(action_num)
```

### DQN

DQN是用神经网络来学习Q值的

网络设计：

```python
网络输入的是当前的状态，输出的是当前状态下每个动作的Q值

故输入神经元个数等于状态的变量个数，输出神经元个数等于动作个数
```

迭代：

```python
迭代Q值的公式与Q-learning一样，Q1‘ = Q1+α*（Reward +γ*Q2 -Q1)

这是公式计算得到的Q1’，但网络输出Q会存在误差，需要反向传播进行学习
```

**NOTE**：

为了使得学习更加平稳，DQN有4个小技巧

- 回放策略
    
    ```python
    DQN并不是对每一步都进行学习，而是有一个固定的经验池
    它会记录自己的执行步骤 或者记录别人的
    当要对网络进行更新学习时，从经验池中随机抽取一步进行学习
    ```
    
- evaluate network 和 target net work
    
    ```python
    由公式Q1‘ = Q1+α*（Reward +γ*Q2 -Q1)
    如果计算Q2的网络 和 要更新的网络属于同一个，那么会使得学习不稳定
    （因为更新网络后，实际上Q2要变化）
    故定义两个一模一样结构的网络
    当状态s来了，输入evaluate network得到各动作Q值
    经过贪心策略得到action,与环境交互得到s'
    将s‘输入target nerwork，计算Q2
    带入公式计算误差，此时Q1是来自evaluate network的 Q2是target network
    当每次要更新网络时，更新evaluate network
    每隔一定时间步，将evaluate network覆盖 target network
    
    个人认为在工程上可以直接计算 网络输出Q1 和 reward+γ*Q2D的误差
    ```
    
- 奖励裁剪
    
    ```python
    在计算reward时，会将reward裁剪到(-1,1)范围，方便适用于其它环境
    ```
    
- 误差计算函数
    
    ```python
    使用Huber函数作为误差计算函数
    当-1<误差<1 , 该函数取平方误差值
    否则，函数取绝对值误差
    这样可以防止误差过大导致的学习不稳定
    ```