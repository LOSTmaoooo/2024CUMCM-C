import torch
import torch.nn as nn
import random
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import time


# 1. 环境：5x5网格，包含障碍物、陷阱和宝藏
class GridWorldEnv:
    def __init__(self):
        self.size = 5  # 5x5网格
        self.start = (0, 0)  # 起点
        self.target = (4, 4)  # 宝藏位置
        self.obstacles = [(1, 1), (1, 2), (3, 3)]  # 固定障碍物
        self.reset()  # 初始化

    def reset(self):
        # 重置环境，随机生成陷阱位置（每次游戏陷阱位置不同）
        self.traps = []
        # 生成2个随机陷阱（不能是起点、宝藏或障碍物）
        while len(self.traps) < 2:
            x = random.randint(0, self.size-1)
            y = random.randint(0, self.size-1)
            pos = (x, y)
            if (pos != self.start and pos != self.target and 
                pos not in self.obstacles and pos not in self.traps):
                self.traps.append(pos)
        
        # 智能体回到起点
        self.x, self.y = self.start
        self.step_count = 0  # 记录步数（防止无限循环）
        
        # 返回状态向量：[智能体x, 智能体y, 陷阱1x, 陷阱1y, 陷阱2x, 陷阱2y]
        return self.get_state()

    def get_state(self):
        # 将状态转换为神经网络可以处理的向量
        return [
            self.x, self.y,
            self.traps[0][0], self.traps[0][1],
            self.traps[1][0], self.traps[1][1]
        ]

    def step(self, action):
        # 初始化done变量，避免未赋值的情况
        done = False
        # 行动：0=上, 1=下, 2=左, 3=右
        new_x, new_y = self.x, self.y
        
        if action == 0:  # 上
            new_y = max(0, new_y - 1)
        elif action == 1:  # 下
            new_y = min(self.size - 1, new_y + 1)
        elif action == 2:  # 左
            new_x = max(0, new_x - 1)
        elif action == 3:  # 右
            new_x = min(self.size - 1, new_x + 1)

        # 检查是否撞到障碍物
        if (new_x, new_y) in self.obstacles:
            # 撞到障碍物，不能移动
            new_x, new_y = self.x, self.y
            reward = -3  # 撞障碍物惩罚
        else:
            # 可以移动，更新位置
            self.x, self.y = new_x, new_y
            self.step_count += 1
            
            # 检查是否到达宝藏
            if (self.x, self.y) == self.target:
                reward = 20  # 找到宝藏奖励
                done = True
            # 检查是否踩到陷阱
            elif (self.x, self.y) in self.traps:
                reward = -10  # 踩陷阱惩罚
                done = False
            # 普通移动
            else:
                reward = -1  # 每步消耗（鼓励走捷径）
                done = False

        # 防止无限循环（超过50步未找到宝藏则结束）
        if self.step_count >= 50:
            done = True
            reward -= 5  # 超时惩罚

        # 返回新状态、奖励和是否结束
        return self.get_state(), reward, done

    def render(self):
        # 可视化网格（可选，用于观察）
        grid = [['.' for _ in range(self.size)] for _ in range(self.size)]
        # 标记障碍物
        for (x, y) in self.obstacles:
            grid[y][x] = '#'
        # 标记陷阱
        for (x, y) in self.traps:
            grid[y][x] = 'X'
        # 标记宝藏
        tx, ty = self.target
        grid[ty][tx] = 'T'
        # 标记智能体
        grid[self.y][self.x] = 'A'
        
        for row in grid:
            print(' '.join(row))
        print('-' * 15)

        print(f"reward: {self.reward}")

# 2. DQN智能体
class DQNAgent:
    def __init__(self):
        self.state_size = 6  # 状态是(x, y)坐标 + 2个陷阱坐标，共6个特征
        self.action_size = 4  # 4个行动方向
        self.memory = deque(maxlen=2000)  # 经验回放池
        self.gamma = 0.9  # 折扣因子
        self.epsilon = 1.0  # 探索率
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99  # 探索率衰减速度
        self.learning_rate = 0.001

        # 神经网络：输入状态，输出4个行动的Q值
        self.model = nn.Sequential(
            nn.Linear(self.state_size, 32),  # 输入层到隐藏层
            nn.ReLU(),
            nn.Linear(32, 16),               # 第二个隐藏层
            nn.ReLU(),
            nn.Linear(16, self.action_size)  # 输出层
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate) # Adam优化器

    def act(self, state):
        # 选择行动：epsilon-greedy策略
        if random.random() < self.epsilon:
            # 随机探索
            return random.choice(range(self.action_size))
        else:
            # 网络决策
            state_tensor = torch.tensor(state, dtype=torch.float32)
            q_values = self.model(state_tensor)
            return torch.argmax(q_values).item()  # 返回Q值最大的行动

    def remember(self, state, action, reward, next_state, done):
        # 存储经验
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        # 从经验池采样并训练
        if len(self.memory) < batch_size:
            return
        
        # 随机采样
        batch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in batch:
            # 转换为张量
            state = torch.tensor(state, dtype=torch.float32)
            next_state = torch.tensor(next_state, dtype=torch.float32)
            reward = torch.tensor(reward, dtype=torch.float32)
            
            # 计算当前Q值
            current_q = self.model(state)[action]
            
            # 计算目标Q值
            if done:
                target_q = reward  # 结束状态没有未来奖励
            else:
                # 非结束状态：目标Q值 = 即时奖励 + 折扣×下一状态的最大Q值
                next_q = torch.max(self.model(next_state))
                target_q = reward + self.gamma * next_q
            
            # 计算损失并更新网络
            loss = nn.MSELoss()(current_q, target_q)
            self.optimizer.zero_grad()  # 清空梯度
            loss.backward()  # 反向传播
            self.optimizer.step()  # 更新参数

        # 衰减探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load_model(self, path="models/dqn_agent.pth",mode=1):
        self.model.load_state_dict(torch.load(path))
        if mode:
            self.model.eval()
        else:
            self.model.train()

    def save_model(self, path="models/dqn_agent.pth"):
        torch.save(self.model.state_dict(), path)

# 3. 训练智能体
def first_train(episodes=800, batch_size=32):
    env = GridWorldEnv()
    agent = DQNAgent()
    scores = []  # 记录每回合的奖励

    for e in range(episodes):
        state = env.reset()  # 现在返回的是包含6个元素的列表
        total_reward = 0
        done = False
        
        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.replay(batch_size)
            
            total_reward += reward
            state = next_state
            
            if done:
                scores.append(total_reward)
                if (e + 1) % 20 == 0:
                    print(f"Episodes: {e+1}/{episodes}, Reward:{total_reward:.2f}, Explore epsilon:{agent.epsilon:.3f}")
                break

    agent.save_model()
    print("Training finished, model saved successfully.")

    # 绘制奖励变化曲线
    plt.plot(scores)
    plt.title('reward over episodes')
    plt.xlabel('episodes')
    plt.ylabel('reward')
    plt.show()
    

def test():
    env = GridWorldEnv()
    agent = DQNAgent()
    agent.load_model(mode=1)
    state = env.reset()
    done = False
    while not done:
        env.render()
        action = agent.act(state)
        print(f"action: {action}")
        next_state, reward, done = env.step(action)
        state = next_state
        time.sleep(1)



if __name__ == "__main__":
    test()
