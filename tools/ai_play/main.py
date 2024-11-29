import requests
from typing import List, Dict, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

action_vocab: Dict[str, int] = {'<UNK>': 0}
action_vocab_rev: Dict[int, str] = {0: '<UNK>'}
action_counter: int = 1
max_actions: int = 1000
panel_vocab: Dict[str, int] = {}
panel_counter: int = 0


class GameEnv:
    """ 游戏环境接口 """

    def __init__(self) -> None:
        self.base_url: str = "http://localhost:5000"

    def reset(self) -> List[int]:
        """
        重置游戏环境
        Return arguments:
        List[int] -- 初始状态
        """
        response = requests.get(f"{self.base_url}/restart")
        result = response.json()
        return result['state']

    def step(self, action: str) -> Tuple[List[int], float, bool]:
        """
        执行动作
        Keyword arguments:
        action -- 动作命令
        Return arguments:
        List[int] -- 下一个状态
        float -- 奖励值
        bool -- 是否结束
        """
        data = {'action': action}
        response = requests.post(f"{self.base_url}/step", json=data)
        result = response.json()
        next_state = result['state']
        reward = result['reward']
        done = result['done']
        return next_state, reward, done

    def get_available_actions(self) -> List[str]:
        """
        获取当前可用的动作列表
        Return arguments:
        List[str] -- 动作列表
        """
        response = requests.get(f"{self.base_url}/actions")
        return response.json()['actions']

    def get_panel_info(self) -> str:
        """
        获取当前面板信息
        Return arguments:
        str -- 当前面板的类型ID
        """
        response = requests.get(f"{self.base_url}/current_panel")
        return response.json()['panel_id']


class PositionalEncoding(nn.Module):
    """ 位置编码 """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        为输入添加位置编码
        Keyword arguments:
        x -- 输入张量，形状为 (batch_size, sequence_length, d_model)
        Return arguments:
        torch.Tensor -- 添加位置编码后的张量，形状为 (batch_size, sequence_length, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class PolicyNetwork(nn.Module):
    """ 策略网络，使用 Transformer 根据动作序列预测下一个动作 """

    def __init__(self, hidden_size: int, action_vocab_size: int, panel_vocab_size: int, action_embed_size: int,
                 num_heads: int, num_layers: int, dropout: float = 0.1) -> None:
        super(PolicyNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.action_embedding = nn.Embedding(action_vocab_size, action_embed_size)
        self.panel_embedding = nn.Embedding(panel_vocab_size, action_embed_size)
        self.positional_encoding = PositionalEncoding(action_embed_size, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=action_embed_size,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.action_head = nn.Linear(action_embed_size, action_vocab_size)

    def forward(self, action_sequence: torch.Tensor, panel_embedding: torch.Tensor) -> torch.Tensor:
        """
        预测下一个动作的 logits
        Keyword arguments:
        action_sequence -- 动作索引的张量，形状为 (sequence_length,)
        panel_embedding -- 当前面板的嵌入向量，形状为 (1, d_model)
        Return arguments:
        torch.Tensor -- 动作 logits 的张量，形状为 (action_vocab_size,)
        """
        if action_sequence.nelement() == 0:
            action_sequence = torch.tensor([0], dtype=torch.long)
        action_embeddings = self.action_embedding(action_sequence)
        action_embeddings = action_embeddings.unsqueeze(0)
        panel_embeddings = panel_embedding.unsqueeze(0)
        combined_input = action_embeddings + panel_embeddings
        combined_input = self.positional_encoding(combined_input)
        transformer_output = self.transformer_encoder(combined_input)
        last_output = transformer_output[:, -1, :]
        logits = self.action_head(last_output)
        logits = logits.squeeze(0)
        return logits


def train() -> None:
    """ 训练策略网络，游戏结束时才进行重置，对过去的策略进行评估，调整优化模型，并保存模型文件 """
    global action_vocab, action_vocab_rev, action_counter, max_actions, panel_vocab, panel_counter
    env = GameEnv()
    num_episodes: int = 1000
    max_sequence_length: int = 100
    hidden_size: int = 128
    action_embed_size: int = 64
    num_heads: int = 8
    num_layers: int = 2
    dropout: float = 0.1
    model_save_path: str = 'policy_model_1.pth'
    policy_net = PolicyNetwork(hidden_size, max_actions, len(panel_vocab), action_embed_size, num_heads, num_layers, dropout)
    if os.path.exists(model_save_path):
        policy_net.load_state_dict(torch.load(model_save_path))
    optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
    entropy_coef: float = 0.01
    episode = 0
    state = env.reset()
    action_sequence: List[int] = []
    log_probs: List[torch.Tensor] = []
    rewards: List[float] = []
    entropies: List[torch.Tensor] = []
    total_reward: float = 0.0
    done: bool = False
    steps: int = 0
    while episode < num_episodes:
        available_actions = env.get_available_actions()
        panel_id = env.get_panel_info()
        if panel_id not in panel_vocab:
            panel_vocab[panel_id] = panel_counter
            panel_counter += 1
        panel_idx = panel_vocab[panel_id]
        for action in available_actions:
            if action not in action_vocab:
                if action_counter < max_actions:
                    action_vocab[action] = action_counter
                    action_vocab_rev[action_counter] = action
                    action_counter += 1
                else:
                    action_vocab[action] = 0
        available_action_indices = [action_vocab.get(action, 0) for action in available_actions]
        action_sequence_tensor = torch.tensor(action_sequence, dtype=torch.long) if action_sequence else torch.empty(0, dtype=torch.long)
        panel_embedding = torch.tensor([panel_idx], dtype=torch.long)
        logits = policy_net(action_sequence_tensor, panel_embedding)
        available_logits = logits[available_action_indices]
        noise = torch.randn_like(available_logits) * 1e-6
        available_logits += noise
        action_probs = F.softmax(available_logits, dim=0)
        action_choice = torch.multinomial(action_probs, 1).item()
        action = available_actions[action_choice]
        next_state, reward, done = env.step(action)
        total_reward += reward
        action_sequence.append(action_vocab.get(action, 0))
        log_probs.append(torch.log(action_probs[action_choice]))
        rewards.append(reward)
        entropy = -torch.sum(action_probs * torch.log(action_probs + 1e-5))
        entropies.append(entropy)
        steps += 1
        if done or steps >= max_sequence_length:
            loss = 0
            for log_prob, reward, entropy in zip(log_probs, rewards, entropies):
                loss += -log_prob * reward
            loss -= entropy_coef * sum(entropies)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            action_sequence = []
            log_probs = []
            rewards = []
            entropies = []
            if done:
                print(f"Episode {episode + 1} finished with total reward {total_reward}")
                total_reward = 0.0
                episode += 1
                state = env.reset()
                steps = 0
    torch.save(policy_net.state_dict(), model_save_path)


if __name__ == '__main__':
    train()

