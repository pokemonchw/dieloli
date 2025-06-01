import requests
from typing import List, Dict, Tuple
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from collections import defaultdict
import math

# —— 超参数 —— #
MAX_ACTIONS = 1000
MAX_PANELS = 512
D_MODEL = 64
NUM_HEADS = 4
NUM_LAYERS = 2
DROPOUT = 0.1
GAMMA = 0.99
ENTROPY_COEF = 0.01
CRITIC_COEF = 0.5
LR = 1e-3
MAX_SEQ_LEN = 50
NUM_EPISODES = 1000
MODEL_PATH = "a2c_transformer_novelty.pth"
NOVELTY_COEF = 0.5
MAX_STEPS_PER_EPISODE = 1000

# —— Vocabulary 映射 —— #
action_vocab: Dict[str, int] = {'<UNK>': 0}
action_counter = 1
panel_vocab: Dict[str, int] = {}
panel_counter = 0

class GameEnv:
    """ 游戏环境接口 """
    def __init__(self) -> None:
        self.base_url = "http://localhost:5000"

    def reset(self) -> List[float]:
        r = requests.get(f"{self.base_url}/restart")
        print(r)
        return r.json()['state']

    def step(self, action: str) -> Tuple[List[float], float, bool]:
        r = requests.post(f"{self.base_url}/step", json={'action': action})
        d = r.json()
        print(r)
        return d['state'], d['reward'], d['done']

    def get_available_actions(self) -> List[str]:
        r = requests.get(f"{self.base_url}/actions")
        print(r)
        return r.json().get('actions', [])

    def get_panel_info(self) -> str:
        r = requests.get(f"{self.base_url}/current_panel")
        print(r)
        return r.json()['panel_id']

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float=0.1, max_len: int=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class ActorCritic(nn.Module):
    """ Actor‑Critic + Transformer (Novelty Bonus 版本) """
    def __init__(self, state_dim: int,
                 action_vocab_size: int, panel_vocab_size: int,
                 d_model: int, num_heads: int, num_layers: int, dropout: float):
        super().__init__()
        # 状态编码
        self.state_embed = nn.Linear(state_dim, d_model)
        # 动作 & 面板 编码
        self.action_embed = nn.Embedding(action_vocab_size, d_model)
        self.panel_embed  = nn.Embedding(panel_vocab_size, d_model)
        # Transformer
        self.pos_enc = PositionalEncoding(d_model, dropout)
        enc_layer = nn.TransformerEncoderLayer(d_model, num_heads,
                                              dropout=dropout,
                                              batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers)
        # 输出头
        self.policy_head = nn.Linear(d_model, action_vocab_size)
        self.value_head  = nn.Linear(d_model, 1)

    def forward(self, state: torch.Tensor,
                      action_seq: torch.Tensor,
                      panel_idx: torch.Tensor
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 状态嵌入
        s_emb = self.state_embed(state)                   # (d_model,)
        # 动作序列嵌入
        if action_seq.numel() == 0:
            action_seq = torch.tensor([0], dtype=torch.long)
        a_emb = self.action_embed(action_seq)             # (L, d_model)
        # 面板嵌入
        p_emb = self.panel_embed(panel_idx).unsqueeze(0)  # (1, d_model)
        # 构造 Transformer 输入：[state] + [action + panel]
        seq = torch.cat([
            s_emb.unsqueeze(0),                          # t=0: state
            a_emb + p_emb.expand(a_emb.size(0), -1)      # t=1..L
        ], dim=0).unsqueeze(0)                            # (1, L+1, d_model)
        seq_enc = self.pos_enc(seq)
        out = self.transformer(seq_enc)                   # (1, L+1, d_model)
        last = out[:, -1, :]
        logits = self.policy_head(last).squeeze(0)        # (action_vocab,)
        value  = self.value_head(last).squeeze(0)         # (1,)
        return logits, value

def train():
    global action_counter, panel_counter

    env = GameEnv()
    init_state = env.reset()
    STATE_DIM = len(init_state)

    net = ActorCritic(
        state_dim=STATE_DIM,
        action_vocab_size=MAX_ACTIONS,
        panel_vocab_size=MAX_PANELS,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    )
    if os.path.exists(MODEL_PATH):
        net.load_state_dict(torch.load(MODEL_PATH))
    opt = optim.Adam(net.parameters(), lr=LR)

    for ep in range(1, NUM_EPISODES + 1):
        state = env.reset()
        action_seq: List[int] = []
        log_probs, values, rewards, entropies = [], [], [], []
        panel_visit_counts = defaultdict(int)
        done = False
        for step in range(1, MAX_STEPS_PER_EPISODE + 1):
            avail = env.get_available_actions()
            if not avail:
                while 1:
                    avail = env.get_available_actions()
                    if avail:
                        break
                    time.sleep(1)
            panel_id = env.get_panel_info()
            panel_visit_counts[panel_id] += 1
            cnt = panel_visit_counts[panel_id]
            novelty_bonus = NOVELTY_COEF / math.sqrt(cnt)

            # 动态扩充 vocab
            if panel_id not in panel_vocab and panel_counter < MAX_PANELS:
                panel_vocab[panel_id] = panel_counter; panel_counter += 1
            for a in avail:
                if a not in action_vocab and action_counter < MAX_ACTIONS:
                    action_vocab[a] = action_counter; action_counter += 1

            avail_idxs = [action_vocab.get(a, 0) for a in avail]
            if not avail_idxs:
                break

            seq_t   = torch.tensor(action_seq[-MAX_SEQ_LEN:], dtype=torch.long)
            state_t = torch.tensor(state, dtype=torch.float32)
            panel_t = torch.tensor(panel_vocab.get(panel_id, 0), dtype=torch.long)

            # 前向 & 采样
            logits, value = net(state_t, seq_t, panel_t)
            probs = F.softmax(logits[avail_idxs], dim=0)
            m = torch.distributions.Categorical(probs)
            sel = m.sample()
            act_idx = avail_idxs[sel.item()]
            action = avail[sel.item()]

            next_state, base_rew, done = env.step(action)
            total_rew = base_rew + novelty_bonus

            # 记录
            log_probs.append(m.log_prob(sel))
            values.append(value)
            rewards.append(total_rew)
            entropies.append(m.entropy())
            action_seq.append(act_idx)
            state = next_state

            if done:
                break

        # 如果这一集没有产生任何 step，直接跳过更新
        if not values:
            print(f"[Episode {ep}] no steps taken, skipping update.")
            continue

        # 计算折扣回报 & Advantage
        R = 0
        returns = []
        for r in reversed(rewards):
            R = r + GAMMA * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32)
        values  = torch.stack(values).squeeze()
        logp    = torch.stack(log_probs)
        ent     = torch.stack(entropies)
        adv     = returns - values

        # 损失
        actor_loss   = -(logp * adv.detach()).mean()
        critic_loss  = adv.pow(2).mean()
        entropy_loss = -ENTROPY_COEF * ent.mean()
        loss = actor_loss + CRITIC_COEF * critic_loss + entropy_loss

        # 更新
        opt.zero_grad()
        loss.backward()
        opt.step()

        if ep % 10 == 0:
            torch.save(net.state_dict(), MODEL_PATH)
            print(f"[Episode {ep}] loss={loss.item():.3f} return={returns.sum():.2f}")

    torch.save(net.state_dict(), MODEL_PATH)

if __name__ == "__main__":
    train()

