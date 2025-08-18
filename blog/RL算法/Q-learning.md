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