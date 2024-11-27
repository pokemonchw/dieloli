import requests
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

class DQN(nn.Module):
    """
    深度Q网络模型，使用全连接神经网络来近似Q值函数。

    参数：
        state_size (int): 状态空间的维度。
        action_size (int): 动作空间的大小。
    """
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )

    def forward(self, x):
        """
        前向传播函数。

        参数：
            x (Tensor): 输入的状态张量。

        返回：
            Tensor: 输出的Q值张量。
        """
        return self.fc(x)


class ReplayBuffer:
    """
    经验回放池，用于存储和采样经验。

    参数：
        capacity (int): 回放池的最大容量。
    """
    def __init__(self, capacity=10000):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """
        存储新的经验。

        参数：
            state (array): 当前状态。
            action (int): 执行动作的索引。
            reward (float): 获得的奖励。
            next_state (array): 下一状态。
            done (bool): 是否结束。
        """
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        随机采样一批经验。

        参数：
            batch_size (int): 采样的批量大小。

        返回：
            list: 采样的经验列表。
        """
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """
        返回当前回放池中经验的数量。

        返回：
            int: 经验数量。
        """
        return len(self.memory)


class GameEnv:
    """
    游戏环境接口，通过HTTP请求与游戏服务器交互。

    方法：
        reset(): 重置游戏，返回初始状态。
        step(action): 执行动作，返回新的状态、奖励和是否结束。
        get_available_actions(): 获取当前可用的命令列表。
        get_state(): 获取当前的世界状态。
        check_rewards(): 检查分数奖励。
    """
    def __init__(self):
        self.base_url = "http://your_game_server_address"  # 替换为实际的游戏服务器地址

    def reset(self):
        """
        重置游戏，获取初始状态。

        返回：
            array: 初始状态的数值向量。
        """
        response = requests.get(f"{self.base_url}/reset")
        state = response.json()['state']
        return self._process_state(state)

    def step(self, action):
        """
        执行动作，获取新的状态、奖励和是否结束。

        参数：
            action (str): 要执行的动作命令。

        返回：
            tuple: (next_state, reward, done)
                - next_state (array): 下一状态的数值向量。
                - reward (float): 获得的奖励。
                - done (bool): 是否结束。
        """
        data = {'action': action}
        response = requests.post(f"{self.base_url}/step", json=data)
        result = response.json()
        next_state = self._process_state(result['state'])
        reward = result['reward']
        done = result['done']
        return next_state, reward, done

    def get_available_actions(self):
        """
        获取当前可用的命令列表。

        返回：
            list: 可用动作命令的列表。
        """
        response = requests.get(f"{self.base_url}/actions")
        return response.json()['actions']

    def get_state(self):
        """
        获取当前的世界状态。

        返回：
            array: 当前状态的数值向量。
        """
        response = requests.get(f"{self.base_url}/state")
        state = response.json()['state']
        return self._process_state(state)

    def check_rewards(self):
        """
        检查当前的分数奖励。

        返回：
            dict: 奖励信息的字典。
        """
        response = requests.get(f"{self.base_url}/rewards")
        return response.json()['rewards']

    def _process_state(self, state):
        """
        处理原始状态数据，将其转换为模型可接受的数值向量。

        参数：
            state (dict): 原始状态数据。

        返回：
            array: 数值向量形式的状态。
        """
        # 假设状态是一个字典，这里需要根据实际情况处理
        state_vector = []
        for key, value in state.items():
            state_vector.append(float(value))
        return np.array(state_vector, dtype=np.float32)


def train():
    """
    训练DQN模型的主函数。

    包括：
        - 初始化环境、模型和优化器。
        - 执行多个训练回合（episodes）。
        - 在每个回合中，与环境交互并学习。
        - 定期更新目标网络。
        - 保存最终模型。
    """
    num_episodes = 500           # 训练的总回合数
    batch_size = 64              # 批量大小
    gamma = 0.99                 # 折扣因子
    epsilon_start = 1.0          # 初始探索率
    epsilon_end = 0.01           # 最小探索率
    epsilon_decay = 500          # 探索率衰减率
    target_update = 10           # 目标网络更新频率
    env = GameEnv()
    state_size = len(env.get_state())             # 状态空间维度
    action_list = env.get_available_actions()     # 获取动作列表
    action_size = len(action_list)                # 动作空间大小
    policy_net = DQN(state_size, action_size)     # 策略网络
    target_net = DQN(state_size, action_size)     # 目标网络
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)  # 优化器
    replay_buffer = ReplayBuffer()
    steps_done = 0
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            # 计算当前的探索率epsilon
            epsilon = epsilon_end + (epsilon_start - epsilon_end) * \
                np.exp(-1. * steps_done / epsilon_decay)
            steps_done += 1
            # 获取当前可用的动作列表和对应的索引
            available_actions = env.get_available_actions()
            action_indices = list(range(len(available_actions)))
            # ε-贪心策略选择动作
            if random.random() > epsilon:
                with torch.no_grad():
                    state_tensor = torch.from_numpy(state).unsqueeze(0)
                    action_values = policy_net(state_tensor)
                    # 只选择当前可用的动作中的最大Q值
                    action_values = action_values[0][action_indices]
                    action_idx = action_indices[torch.argmax(action_values).item()]
            else:
                action_idx = random.choice(action_indices)
            action = available_actions[action_idx]  # 获取实际的动作命令
            next_state, reward, done = env.step(action)
            total_reward += reward
            # 存储经验
            replay_buffer.push(state, action_idx, reward, next_state, done)
            state = next_state
            # 当经验足够时开始训练
            if len(replay_buffer) > batch_size:
                transitions = replay_buffer.sample(batch_size)
                batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*transitions)
                batch_state = torch.tensor(batch_state)
                batch_action = torch.tensor(batch_action, dtype=torch.long)
                batch_reward = torch.tensor(batch_reward)
                batch_next_state = torch.tensor(batch_next_state)
                batch_done = torch.tensor(batch_done, dtype=torch.float32)
                # 计算当前Q值
                q_values = policy_net(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
                # 计算下一个状态的最大Q值
                next_q_values = target_net(batch_next_state).max(1)[0]
                # 计算目标Q值
                expected_q_values = batch_reward + gamma * next_q_values * (1 - batch_done)
                # 计算损失
                loss = nn.MSELoss()(q_values, expected_q_values.detach())
                # 更新模型参数
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        # 更新目标网络
        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())
        print(f"Episode {episode}, Total Reward: {total_reward}")
    # 保存模型
    torch.save(policy_net.state_dict(), "dqn_model.pth")


if __name__ == "__main__":
    train()

