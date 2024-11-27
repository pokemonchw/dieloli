import requests
from typing import List, Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

action_vocab: Dict[str, int] = {'<UNK>': 0}
action_vocab_rev: Dict[int, str] = {0: '<UNK>'}
action_counter: int = 1
max_actions: int = 1000


class GameEnv:
    """
    游戏环境接口
    """

    def __init__(self) -> None:
        self.base_url: str = "http://localhost:5000"

    def reset(self) -> List[int]:
        """
        重置游戏环境
        Return arguments:
        List[str] -- 初始状态
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


class PositionalEncoding(nn.Module):
    """
    位置编码，用于为输入的嵌入添加位置信息
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        为输入添加位置编码。
        Keyword arguments:
        x -- 输入张量，形状为 (sequence_length, batch_size, d_model)
        Return arguments:
        x -- 添加位置编码后的张量，形状为 (sequence_length, batch_size, d_model)
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class PolicyNetwork(nn.Module):
    """
    策略网络，使用 Transformer 根据动作序列预测下一个动作
    """

    def __init__(self, hidden_size: int, action_vocab_size: int, action_embed_size: int,
                 num_heads: int, num_layers: int, dropout: float = 0.1) -> None:
        super(PolicyNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.action_embedding = nn.Embedding(action_vocab_size, action_embed_size)
        self.positional_encoding = PositionalEncoding(action_embed_size, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=action_embed_size, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.action_head = nn.Linear(action_embed_size, action_vocab_size)

    def forward(self, action_sequence: torch.Tensor) -> torch.Tensor:
        """
        预测下一个动作的 logits
        Keyword arguments:
        action_sequence -- 动作索引的张量，形状为 (sequence_length,)
        Return arguments:
        logits -- 动作 logits 的张量，形状为 (action_vocab_size,)
        """
        if action_sequence.nelement() == 0:
            action_sequence = torch.tensor([0], dtype=torch.long)
        action_embeddings = self.action_embedding(action_sequence)
        action_embeddings = action_embeddings.unsqueeze(1)
        action_embeddings = self.positional_encoding(action_embeddings)
        transformer_output = self.transformer_encoder(action_embeddings)
        last_output = transformer_output[-1, 0, :]
        logits = self.action_head(last_output)
        return logits


def train() -> None:
    """
    训练策略网络，使用 Transformer，并避免重复相同的动作
    """
    global action_vocab, action_vocab_rev, action_counter, max_actions
    env = GameEnv()
    num_episodes: int = 1000
    max_sequence_length: int = 50
    hidden_size: int = 128
    action_embed_size: int = 64
    num_heads: int = 8
    num_layers: int = 2
    dropout: float = 0.1
    policy_net = PolicyNetwork(hidden_size, max_actions, action_embed_size, num_heads, num_layers, dropout)
    optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
    entropy_coef: float = 0.01
    for episode in range(num_episodes):
        state = env.reset()
        action_sequence: List[int] = []
        log_probs: List[torch.Tensor] = []
        rewards: List[float] = []
        entropies: List[torch.Tensor] = []
        total_reward: float = 0.0
        done: bool = False
        steps: int = 0
        while not done and steps < max_sequence_length:
            available_actions = env.get_available_actions()
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
            logits = policy_net(action_sequence_tensor)
            available_logits = logits[available_action_indices]
            noise = torch.randn_like(available_logits) * 1e-6
            available_logits += noise
            action_probs = F.softmax(available_logits, dim=0)
            entropy = -torch.sum(action_probs * torch.log(action_probs + 1e-10))
            entropies.append(entropy)
            m = torch.distributions.Categorical(action_probs)
            action_idx_in_available = m.sample()
            log_prob = m.log_prob(action_idx_in_available)
            log_probs.append(log_prob)
            action_idx = available_action_indices[action_idx_in_available.item()]
            action = action_vocab_rev.get(action_idx, '<UNK>')
            if len(action_sequence) >= 1 and action_idx == action_sequence[-1]:
                attempt = 0
                max_attempts = 10
                while action_idx == action_sequence[-1] and attempt < max_attempts:
                    action_idx_in_available = m.sample()
                    log_prob = m.log_prob(action_idx_in_available)
                    action_idx = available_action_indices[action_idx_in_available.item()]
                    action = action_vocab_rev.get(action_idx, '<UNK>')
                    attempt += 1
                if attempt == max_attempts:
                    sorted_probs, sorted_indices = torch.sort(action_probs, descending=True)
                    for idx in sorted_indices:
                        temp_action_idx = available_action_indices[idx.item()]
                        if temp_action_idx != action_sequence[-1]:
                            action_idx = temp_action_idx
                            log_prob = torch.log(action_probs[idx])
                            break
            next_state, reward, done = env.step(action)
            rewards.append(reward)
            total_reward += reward
            steps += 1
            action_sequence.append(action_idx)
        total_return = sum(rewards)
        policy_loss = -torch.stack(log_probs).sum() * total_return
        entropy_loss = -entropy_coef * torch.stack(entropies).sum()
        loss = policy_loss + entropy_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Episode {episode}, Total Reward: {total_reward}")
        if entropy_loss.item() < (entropy_coef * 0.5):
            entropy_coef *= 0.9
        elif entropy_loss.item() > (entropy_coef * 1.5):
            entropy_coef *= 1.1


if __name__ == '__main__':
    train()

