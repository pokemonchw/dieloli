import requests
from typing import List, Dict, Tuple
import torch
import torch.nn as nn
import os
import argparse
from Script.Core import game_path_config, cache_control, game_type

cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """
action_vocab: Dict[str, int] = {'<UNK>': 0}
action_vocab_rev: Dict[int, str] = {0: '<UNK>'}
panel_vocab: Dict[str, int] = {'<UNK>': 0}

max_actions: int = 1000  # 用于定义 action_embedding 的大小
max_panels: int = 1      # 用于定义 panel_embedding 的大小

# ----------------------------
# 游戏环境封装（不变）
# ----------------------------
class GameEnv:
    def __init__(self) -> None:
        self.base_url: str = "http://localhost:5000"

    def reset(self) -> List[int]:
        resp = requests.get(f"{self.base_url}/restart")
        return resp.json()['state']

    def step(self, action: str) -> Tuple[List[int], float, bool]:
        resp = requests.post(f"{self.base_url}/step", json={'action': action})
        data = resp.json()
        return data['state'], data['reward'], data['done']

    def get_available_actions(self) -> List[str]:
        resp = requests.get(f"{self.base_url}/actions")
        actions = resp.json().get('actions', [])
        return actions if actions else ['']

    def get_panel_info(self) -> str:
        resp = requests.get(f"{self.base_url}/current_panel")
        return resp.json().get('panel_id', '')

# ----------------------------
# 模型定义：恢复原名 key
# ----------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class PolicyNetwork(nn.Module):
    def __init__(
        self, action_vocab_size: int, panel_vocab_size: int,
        action_embed_size: int = 64, num_heads: int = 8,
        num_layers: int = 2, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.action_embedding = nn.Embedding(action_vocab_size, action_embed_size)
        self.panel_embedding  = nn.Embedding(panel_vocab_size, action_embed_size)
        # 恢复成 checkpoint 里用的名字
        self.positional_encoding = PositionalEncoding(action_embed_size, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=action_embed_size, nhead=num_heads,
            dropout=dropout, batch_first=True
        )
        # 恢复成 transformer_encoder
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.action_head = nn.Linear(action_embed_size, action_vocab_size)

    def forward(self, action_seq: torch.Tensor, panel_idx: torch.Tensor) -> torch.Tensor:
        if action_seq.nelement() == 0:
            action_seq = torch.tensor([0], dtype=torch.long)
        ae = self.action_embedding(action_seq).unsqueeze(0)    # (1, L, E)
        pe = self.panel_embedding(panel_idx).unsqueeze(1)      # (1, 1, E)
        x = ae + pe                                           # (1, L, E)
        x = self.positional_encoding(x)
        out = self.transformer_encoder(x)                     # (1, L, E)
        logits = self.action_head(out[:, -1, :])              # (1, V)
        return logits.squeeze(0)                              # (V,)

# ----------------------------
# 交互逻辑：加载模型时 strict=False
# ----------------------------
def run_interactive() -> None:
    model_path = game_path_config.AI_MODEL_PATH
    env = GameEnv()
    policy = PolicyNetwork(
        action_vocab_size=max_actions,
        panel_vocab_size=max_panels,
    )

    if os.path.exists(model_path):
        ckpt = torch.load(model_path, map_location='cpu')
        ckpt.pop('panel_embedding.weight', None)
        load_info = policy.load_state_dict(ckpt, strict=False)
        print(f"模型已加载（部分权重）：\n  missing_keys={load_info.missing_keys}\n  unexpected_keys={load_info.unexpected_keys}")
    else:
        print(f"警告：模型文件 {model_path} 不存在，使用随机初始化权重。")
    policy.eval()

    action_sequence: List[int] = []

    while cache.observe_switch:
        actions = env.get_available_actions()
        panel_id = env.get_panel_info()

        panel_idx = torch.tensor([panel_vocab.get(panel_id, 0)], dtype=torch.long)
        seq_tensor = torch.tensor(action_sequence, dtype=torch.long)

        with torch.no_grad():
            logits = policy(seq_tensor, panel_idx)
            avail_idxs = [action_vocab.get(a, 0) for a in actions]
            avail_logits = logits[avail_idxs]
            choice = torch.argmax(avail_logits).item()

        action = actions[choice]
        next_state, reward, done = env.step(action)
        action_sequence.append(action_vocab.get(action, 0))
