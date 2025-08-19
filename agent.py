import torch
import torch.nn as nn
import random
from collections import deque
import numpy as np

class DQNAgent:
    def __init__(self,state_dim,num_plots=3,num_crops=4):
        self.state_dim = state_dim # 状态维度：3地块×4特征 + 1时间 = 13
        self.num_plots = num_plots
        self.num_crops = num_crops # 4种作物（+不种，共5个选择）
        self.epsilon = 1.0 # 初始探索率
        self.memory = deque(maxlen=1000) # 经验回放池
        self.gamma = 0.9 # 折扣因子

        # 网络：输入状态，输出每个地块的选择Q值、
        self.model= nn.Sequential(
            nn.Linear(self.state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, self.num_plots * (self.num_crops + 1))
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def select_action(self, state):
        action=[]
        if random.random()<self.epsilon:
            # 随机行动
            for plot_idx in range(3):
                pass #action.append(random.randint(0,4))
        else:
            # 网络决策
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q_values = self.model(state).view(self.num_plots, self.num_crops + 1)

            for plot_idx in range(3):
                pass

        return action
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self, batch_size=16):
        if len(self.memory) < batch_size:
            return
        batch=random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in batch:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
            reward = torch.tensor(reward, dtype=torch.float32)

            # 当前Q值：行动对应的Q值之和
            current_q = self.model(state).view(self.num_plots, self.num_crops + 1)
            total_current_q = 0
            for plot_idx in range(3):
                a = action[plot_idx]
                # 行动转为索引（-1→3，0→0，1→1，2→2）
                a_idx = a + 1 if a == -1 else a
                total_current_q += current_q[plot_idx, a_idx]

            # 目标Q值
            next_q = self.model(next_state).view(self.num_plots, self.num_crops + 1)
            max_next_q = next_q.max(dim=1).values.sum()  # 下一状态最大Q值之和
            target_q = reward + (1 - done) * self.gamma * max_next_q

            # 更新网络
            loss = nn.MSELoss()(total_current_q, target_q)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # 衰减探索率
        self.epsilon = max(0.01, self.epsilon * 0.99)
