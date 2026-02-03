import os
import math
import time
import random
import re
from collections import defaultdict, deque
from typing import List, Dict, Tuple, Optional

import requests
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# ==================== 超参数定义 ====================

MAX_PANELS = 512
""" 最大面板数量 """
MAX_VERBS = 64
""" 最大动词数量 """
MAX_TARGETS = 256
""" 最大目标数量 """

LATENT_DIM = 256
""" 潜在空间维度 """
HIDDEN_DIM = 512
""" 隐藏层维度 """
STATE_DIM = None
""" 状态维度（运行时确定） """
PANEL_EMBED_DIM = 64
""" 面板嵌入维度 """
VERB_EMBED_DIM = 32
""" 动词嵌入维度 """
TARGET_EMBED_DIM = 32
""" 目标嵌入维度 """

PLAN_HORIZON = 10
""" 规划时间步长 """
CEM_ITERATIONS = 6
""" 交叉熵方法迭代次数 """
CEM_POPULATION = 256
""" CEM种群大小 """
CEM_ELITE_RATIO = 0.15
""" CEM精英比例 """
CEM_TEMP = 1.2
""" CEM温度参数 """

EXIT_ACTION_PRIOR = 5.0
""" 退出动作的先验权重 """
NOOP_ACTION_PRIOR = 0.3
""" 无操作动作的先验权重 """

GAMMA = 0.98
""" 折扣因子 """
SEMANTIC_LOOP_PENALTY = 2.0
""" 语义循环惩罚 """
STATE_LOOP_PENALTY = 0.8
""" 状态循环惩罚 """
STUCK_PENALTY = 0.5
""" 卡住惩罚 """

LR = 2e-4
""" 学习率 """
BATCH_SIZE = 32
""" 批次大小 """
SEQ_LEN = 8
""" 序列长度 """
REPLAY_CAPACITY = 20000
""" 经验回放容量 """
WARMUP_STEPS = 800
""" 预热步数 """
TRAIN_INTERVAL = 4
""" 训练间隔 """
TRAIN_EPOCHS = 3
""" 每次训练的轮数 """

NUM_EPISODES = 1000
""" 总训练回合数 """
MAX_STEPS_PER_EPISODE = 1000000
""" 每回合最大步数 """
MODEL_PATH = "world_model_structured.pth"
""" 模型保存路径 """

W_REWARD = 1.0
""" 奖励损失权重 """
W_DONE = 1.0
""" 完成损失权重 """
W_PANEL = 3.0
""" 面板损失权重 """
W_RECON = 0.3
""" 重建损失权重 """
W_KL = 0.05
""" KL散度损失权重 """

# ==================== 词汇表定义 ====================

panel_vocab: Dict[str, int] = {"<UNK>": 0}
""" 面板词汇表 """
panel_counter = 1
""" 面板计数器 """

verb_vocab: Dict[str, int] = {"<UNK>": 0, "<PAD>": 1}
""" 动词词汇表 """
verb_counter = 2
""" 动词计数器 """

target_vocab: Dict[str, int] = {"<UNK>": 0, "<PAD>": 1}
""" 目标词汇表 """
target_counter = 2
""" 目标计数器 """

EXIT_VERBS = {"exit", "return", "back", "close", "cancel", "返回", "退出", "关闭"}
""" 退出类动词集合 """
NOOP_VERBS = {"view", "check", "查看", "检查"}
""" 无操作类动词集合 """


# ==================== 工具函数 ====================

def parse_action(action: str) -> Tuple[str, str]:
    """
    解析动作字符串为动词和目标
    Keyword arguments:
    action -- 动作字符串，格式为 "verb_target"
    Return arguments:
    Tuple[str, str] -- (动词, 目标)，如果没有目标则返回"<PAD>"
    """
    parts = action.split("_", 1)
    if len(parts) == 1:
        return parts[0], "<PAD>"
    return parts[0], parts[1]


def get_action_type(verb: str) -> str:
    """
    根据动词分类动作类型
    Keyword arguments:
    verb -- 动词字符串
    Return arguments:
    str -- 动作类型: "EXIT"（退出）、"NOOP"（无操作）或"NORMAL"（普通）
    """
    v_lower = verb.lower()
    if v_lower in EXIT_VERBS:
        return "EXIT"
    if v_lower in NOOP_VERBS:
        return "NOOP"
    return "NORMAL"


def get_action_indices(action: str) -> Tuple[int, int]:
    """
    获取动作对应的词汇表索引，如果不存在则添加到词汇表
    Keyword arguments:
    action -- 动作字符串
    Return arguments:
    Tuple[int, int] -- (动词索引, 目标索引)
    """
    global verb_counter, target_counter
    verb, target = parse_action(action)
    if verb not in verb_vocab and verb_counter < MAX_VERBS:
        verb_vocab[verb] = verb_counter
        verb_counter += 1
    if target not in target_vocab and target_counter < MAX_TARGETS:
        target_vocab[target] = target_counter
        target_counter += 1
    verb_idx = verb_vocab.get(verb, 0)
    target_idx = target_vocab.get(target, 0)
    return verb_idx, target_idx


# ==================== 游戏环境类 ====================

class GameEnv:
    """游戏环境接口类，通过HTTP与游戏服务器交互"""

    def __init__(self) -> None:
        """
        初始化游戏环境
        """
        self.base_url = "http://localhost:5000"
        """ 游戏服务器基础URL """

    def reset(self) -> List[float]:
        """
        重置游戏环境到初始状态
        Return arguments:
        List[float] -- 初始状态向量
        """
        os.system("pkill game.py")
        os.system("./restart.sh")
        time.sleep(10)
        r = requests.post(f"{self.base_url}/step", json={"action": "1"})
        time.sleep(2)
        r = requests.post(f"{self.base_url}/step", json={"action": "0"})
        time.sleep(2)
        r = requests.post(f"{self.base_url}/step", json={"action": "0"})
        d = r.json()
        return d["state"]

    def step(self, action: str) -> Tuple[List[float], float, bool]:
        """
        执行一步动作
        Keyword arguments:
        action -- 要执行的动作字符串
        Return arguments:
        Tuple[List[float], float, bool] -- (下一状态, 奖励, 是否结束)
        """
        r = requests.post(f"{self.base_url}/step", json={"action": action})
        print(r)
        d = r.json()
        return d["state"], d["reward"], d["done"]

    def get_available_actions(self) -> List[str]:
        """
        获取当前可用的动作列表
        Return arguments:
        List[str] -- 可用动作列表
        """
        r = requests.get(f"{self.base_url}/actions")
        return r.json().get("actions", [])

    def get_panel_info(self) -> str:
        """
        获取当前面板ID
        Return arguments:
        str -- 当前面板的ID字符串
        """
        r = requests.get(f"{self.base_url}/current_panel")
        return r.json()["panel_id"]

class TopologicalMap:
    """ 拓扑图管理器 """

    def __init__(self):
        self.nodes = defaultdict(int)
        """ 节点列表 """
        self.edges = defaultdict(set)
        """ 路径集合 """
        self.frontier_actions = defaultdict(set)
        """ 节点下的行动集合 """

    def update(self, prev_panel: int, action: str, current_panel: int, available_actions: List[str]):
        """
        更新节点的路径信息
        Keyword arguments:
        prev_panel -- 前一个面板
        action -- 执行的行动
        current_panel -- 当前面板
        available_actions -- 当前面板可用的行动表
        """
        self.nodes[current_panel] += 1
        if prev_panel is not None:
            self.edges[(prev_panel, action)].add(current_panel)
        if current_panel not in self.frontier_actions:
            self.frontier_actions[current_panel] = set(available_actions)
        if action in self.frontier_actions[prev_panel]:
            self.frontier_actions[prev_panel].remove(action)

    def get_intrinsic_reward(self, panel_id: int) -> float:
        """
        获取探索面板奖励
        Keyword arguments:
        panel_id -- 面板id
        Return arguments:
        float -- 奖励积分
        """
        count = self.nodes.get(panel_id, 0)
        return 1.0 / math.sqrt(count + 1)


# ==================== 经验回放缓冲区类 ====================

class SequenceReplayBuffer:
    """序列经验回放缓冲区，用于存储和采样序列数据"""

    def __init__(self, capacity: int, seq_len: int):
        """
        初始化序列经验回放缓冲区
        Keyword arguments:
        capacity -- 缓冲区容量
        seq_len -- 序列长度
        """
        self.capacity = capacity
        """ 缓冲区最大容量 """
        self.seq_len = seq_len
        """ 每个序列的长度 """
        self.sequences = deque(maxlen=capacity)
        """ 存储序列的双端队列 """

    def push_episode(self, episode_data: List[Dict]):
        """
        将一个回合的数据切分为序列并存入缓冲区
        Keyword arguments:
        episode_data -- 回合数据列表，每个元素是一个包含状态、动作等信息的字典
        """
        if len(episode_data) < self.seq_len:
            return
        for i in range(len(episode_data) - self.seq_len + 1):
            seq = episode_data[i:i+self.seq_len]
            self.sequences.append(seq)

    def sample(self, batch_size: int):
        """
        从缓冲区中随机采样一个批次的序列
        Keyword arguments:
        batch_size -- 批次大小
        Return arguments:
        Dict -- 包含states、verbs、targets等键的字典，值为torch.Tensor
        """
        if len(self.sequences) < batch_size:
            return None
        batch_seqs = random.sample(self.sequences, batch_size)
        states = []
        verbs = []
        targets = []
        rewards = []
        panels = []
        dones = []
        for seq in batch_seqs:
            states.append([step['state'] for step in seq])
            verbs.append([step['verb_idx'] for step in seq])
            targets.append([step['target_idx'] for step in seq])
            rewards.append([step['reward'] for step in seq])
            panels.append([step['panel_idx'] for step in seq])
            dones.append([step['done'] for step in seq])
        return {
            'states': torch.tensor(states, dtype=torch.float32),
            'verbs': torch.tensor(verbs, dtype=torch.long),
            'targets': torch.tensor(targets, dtype=torch.long),
            'rewards': torch.tensor(rewards, dtype=torch.float32),
            'panels': torch.tensor(panels, dtype=torch.long),
            'dones': torch.tensor(dones, dtype=torch.float32),
        }

    def __len__(self):
        """
        返回缓冲区中序列的数量
        Return arguments:
        int -- 序列数量
        """
        return len(self.sequences)


# ==================== 世界模型类 ====================

class PanelConditionedWorldModel(nn.Module):
    """面板条件世界模型，使用变分自编码器和GRU预测环境动态"""

    def __init__(self, state_dim: int):
        """
        初始化面板条件世界模型
        Keyword arguments:
        state_dim -- 状态向量的维度
        """
        super().__init__()
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )
        """ 状态编码器网络 """
        self.panel_embed = nn.Embedding(MAX_PANELS, PANEL_EMBED_DIM)
        """ 面板嵌入层 """
        self.verb_embed = nn.Embedding(MAX_VERBS, VERB_EMBED_DIM)
        """ 动词嵌入层 """
        self.target_embed = nn.Embedding(MAX_TARGETS, TARGET_EMBED_DIM)
        """ 目标嵌入层 """
        action_dim = VERB_EMBED_DIM + TARGET_EMBED_DIM
        self.gru = nn.GRUCell(
            LATENT_DIM + action_dim + PANEL_EMBED_DIM,
            HIDDEN_DIM
        )
        """ GRU循环单元，用于状态转移 """
        self.posterior_net = nn.Sequential(
            nn.Linear(HIDDEN_DIM + 128, 256),
            nn.ReLU(),
            nn.Linear(256, LATENT_DIM * 2)
        )
        """ 后验网络，输出潜在变量的均值和对数标准差 """
        self.prior_net = nn.Sequential(
            nn.Linear(HIDDEN_DIM, 256),
            nn.ReLU(),
            nn.Linear(256, LATENT_DIM * 2)
        )
        """ 先验网络，输出潜在变量的均值和对数标准差 """
        self.reward_head = nn.Sequential(
            nn.Linear(HIDDEN_DIM + LATENT_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        """ 奖励预测头 """
        self.done_head = nn.Sequential(
            nn.Linear(HIDDEN_DIM + LATENT_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        """ 完成标志预测头 """
        self.panel_head = nn.Sequential(
            nn.Linear(HIDDEN_DIM + LATENT_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, MAX_PANELS)
        )
        """ 面板预测头 """
        self.state_decoder = nn.Sequential(
            nn.Linear(HIDDEN_DIM + LATENT_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, state_dim)
        )
        """ 状态解码器网络 """

    def initial_state(self, batch_size: int, device: torch.device):
        """
        初始化隐藏状态和潜在变量
        Keyword arguments:
        batch_size -- 批次大小
        device -- 设备（CPU或CUDA）
        Return arguments:
        Tuple[torch.Tensor, torch.Tensor] -- (隐藏状态h, 潜在变量z)
        """
        h = torch.zeros(batch_size, HIDDEN_DIM, device=device)
        z = torch.zeros(batch_size, LATENT_DIM, device=device)
        return h, z

    def encode(self, state: torch.Tensor):
        """
        编码状态向量
        Keyword arguments:
        state -- 状态张量
        Return arguments:
        torch.Tensor -- 编码后的状态表示
        """
        return self.state_encoder(state)

    def sample_latent(self, mean: torch.Tensor, logstd: torch.Tensor):
        """
        从高斯分布中采样潜在变量（重参数化技巧）
        Keyword arguments:
        mean -- 均值张量
        logstd -- 对数标准差张量
        Return arguments:
        torch.Tensor -- 采样得到的潜在变量
        """
        std = torch.exp(logstd)
        eps = torch.randn_like(mean)
        return mean + std * eps

    def compose_action(self, verb_idx: torch.Tensor, target_idx: torch.Tensor):
        """
        组合动词和目标的嵌入表示
        Keyword arguments:
        verb_idx -- 动词索引张量
        target_idx -- 目标索引张量
        Return arguments:
        torch.Tensor -- 拼接后的动作嵌入
        """
        v_emb = self.verb_embed(verb_idx)
        t_emb = self.target_embed(target_idx)
        return torch.cat([v_emb, t_emb], dim=-1)

    def transition(self, h, z, verb_idx, target_idx, panel_idx):
        """
        状态转移函数，根据当前状态和动作预测下一个隐藏状态
        Keyword arguments:
        h -- 当前隐藏状态
        z -- 当前潜在变量
        verb_idx -- 动词索引
        target_idx -- 目标索引
        panel_idx -- 面板索引
        Return arguments:
        torch.Tensor -- 下一个隐藏状态
        """
        a_emb = self.compose_action(verb_idx, target_idx)
        p_emb = self.panel_embed(panel_idx)
        inp = torch.cat([z, a_emb, p_emb], dim=-1)
        h_next = self.gru(inp, h)
        return h_next

    def posterior(self, h, obs_embed):
        """
        计算后验分布参数（给定观测）
        Keyword arguments:
        h -- 隐藏状态
        obs_embed -- 观测嵌入
        Return arguments:
        Tuple[torch.Tensor, torch.Tensor] -- (均值, 对数标准差)
        """
        inp = torch.cat([h, obs_embed], dim=-1)
        out = self.posterior_net(inp)
        mean, logstd = torch.chunk(out, 2, dim=-1)
        logstd = torch.clamp(logstd, -10, 2)
        return mean, logstd

    def prior(self, h):
        """
        计算先验分布参数
        Keyword arguments:
        h -- 隐藏状态
        Return arguments:
        Tuple[torch.Tensor, torch.Tensor] -- (均值, 对数标准差)
        """
        out = self.prior_net(h)
        mean, logstd = torch.chunk(out, 2, dim=-1)
        logstd = torch.clamp(logstd, -10, 2)
        return mean, logstd

    def predict(self, h, z):
        """
        根据隐藏状态和潜在变量预测奖励、完成标志、面板和状态
        Keyword arguments:
        h -- 隐藏状态
        z -- 潜在变量
        Return arguments:
        Tuple -- (奖励预测, 完成预测, 面板logits, 状态重建)
        """
        hz = torch.cat([h, z], dim=-1)
        r = self.reward_head(hz).squeeze(-1)
        d = torch.sigmoid(self.done_head(hz)).squeeze(-1)
        p = self.panel_head(hz)
        s = self.state_decoder(hz)
        return r, d, p, s

    def forward_seq(self, states, verbs, targets, panels):
        """
        前向传播处理序列数据
        Keyword arguments:
        states -- 状态序列张量 [B, T, D]
        verbs -- 动词序列张量 [B, T]
        targets -- 目标序列张量 [B, T]
        panels -- 面板序列张量 [B, T]
        Return arguments:
        Dict -- 包含预测结果和分布参数的字典
        """
        B, T, D = states.shape
        device = states.device
        h, z = self.initial_state(B, device)
        outputs = {
            'rewards': [],
            'dones': [],
            'panel_logits': [],
            'state_recons': [],
            'post_means': [],
            'post_logstds': [],
            'prior_means': [],
            'prior_logstds': []
        }
        for t in range(T):
            obs_embed = self.encode(states[:, t])
            post_mean, post_logstd = self.posterior(h, obs_embed)
            z = self.sample_latent(post_mean, post_logstd)
            prior_mean, prior_logstd = self.prior(h)
            r, d, p_logits, s_recon = self.predict(h, z)
            outputs['rewards'].append(r)
            outputs['dones'].append(d)
            outputs['panel_logits'].append(p_logits)
            outputs['state_recons'].append(s_recon)
            outputs['post_means'].append(post_mean)
            outputs['post_logstds'].append(post_logstd)
            outputs['prior_means'].append(prior_mean)
            outputs['prior_logstds'].append(prior_logstd)
            if t < T - 1:
                h = self.transition(h, z, verbs[:, t], targets[:, t], panels[:, t])
        for k in ['rewards', 'dones', 'state_recons']:
            outputs[k] = torch.stack(outputs[k], dim=1)
        outputs['panel_logits'] = torch.stack(outputs['panel_logits'], dim=1)
        for k in ['post_means', 'post_logstds', 'prior_means', 'prior_logstds']:
            outputs[k] = torch.stack(outputs[k], dim=1)
        return outputs

    def imagine_step(self, h, z, verb_idx, target_idx, panel_idx):
        """
        在想象空间中执行一步模拟
        Keyword arguments:
        h -- 当前隐藏状态
        z -- 当前潜在变量
        verb_idx -- 动词索引
        target_idx -- 目标索引
        panel_idx -- 面板索引
        Return arguments:
        Tuple -- (下一隐藏状态, 下一潜在变量, 奖励预测, 完成预测, 面板logits)
        """
        h_next = self.transition(h, z, verb_idx, target_idx, panel_idx)
        prior_mean, prior_logstd = self.prior(h_next)
        z_next = self.sample_latent(prior_mean, prior_logstd)
        r, d, p_logits, _ = self.predict(h_next, z_next)
        return h_next, z_next, r, d, p_logits


# ==================== 结构化CEM规划器类 ====================

class StructuredCEMEnhancedTopologicalPlanner:
    """使用交叉熵方法(CEM)结合拓扑图的结构化规划器"""

    def __init__(self, model: PanelConditionedWorldModel, topo_map: TopologicalMap, intrinsic_weight: float = 0.5, frontier_weight: float = 0.3, known_loop_penalty: float = 1.0):
        """
        初始化规划器
        Keyword arguments:
        model -- 世界模型实例
        topo_map -- 拓扑图
        intrinsic_weight -- 面板新颖度奖励修正
        frontier_weight -- 动作新颖度奖励修正
        known_loop_penalty -- 动作连通性惩罚
        """
        self.model = model
        """ 世界模型 """
        self.topo_map = topo_map
        """ 拓扑图 """
        self.intrinsic_weight = intrinsic_weight
        """ 面板新颖度奖励修正 """
        self.frontier_weight = frontier_weight
        """ 动作新颖度奖励修正 """
        self.known_loop_penalty = known_loop_penalty
        """ 动作连通性惩罚 """

    @staticmethod
    def _panel_id_from_idx(panel_idx: int) -> str:
        """
        将 panel index 映射回 panel_id（字符串）。
        若未知则返回一个稳定占位符，确保 topo_map 仍能计数。
        """
        for k, v in panel_vocab.items():
            if v == panel_idx:
                return k
        return f"<P{panel_idx}>"

    def plan(self, h: torch.Tensor, z: torch.Tensor, panel_idx: torch.Tensor, available_actions, horizon=PLAN_HORIZON):
        """
        使用CEM规划最优动作序列
        Keyword arguments:
        h -- 当前隐藏状态
        z -- 当前潜在变量
        panel_idx -- 当前面板索引
        available_actions -- 可用动作列表
        horizon -- 规划时间步长
        Return arguments:
        str -- 最优动作字符串，如果无可用动作则返回None
        """
        if not available_actions:
            return None
        device = h.device
        num_actions = len(available_actions)
        # 初始化动作先验
        priors = torch.ones(num_actions, device=device)
        for i, (_, _, _, atype) in enumerate(available_actions):
            if atype == "EXIT":
                priors[i] = EXIT_ACTION_PRIOR
            elif atype == "NOOP":
                priors[i] = NOOP_ACTION_PRIOR
        # 初始化分布
        dist = [priors / priors.sum() for _ in range(horizon)]
        best_sequence = None
        best_score = -1e9
        # CEM迭代
        for iteration in range(CEM_ITERATIONS):
            sequences = []
            # 采样种群
            for _ in range(CEM_POPULATION):
                seq = []
                for t in range(horizon):
                    probs = dist[t]
                    if CEM_TEMP != 1.0:
                        probs = torch.pow(probs, 1.0 / CEM_TEMP)
                        probs = probs / probs.sum()
                    idx = torch.multinomial(probs, 1).item()
                    seq.append(idx)
                sequences.append(seq)
            # 评估序列
            scores = []
            for seq in sequences:
                score = self._evaluate_sequence(h, z, panel_idx, seq, available_actions)
                scores.append(score)
            # 更新最佳序列
            scores_t = torch.tensor(scores, device=device)
            max_idx = torch.argmax(scores_t).item()
            if scores[max_idx] > best_score:
                best_score = scores[max_idx]
                best_sequence = sequences[max_idx]
            # 选择精英并更新分布
            num_elite = max(1, int(CEM_POPULATION * CEM_ELITE_RATIO))
            elite_indices = torch.topk(scores_t, num_elite).indices
            for t in range(horizon):
                counts = torch.zeros(num_actions, device=device)
                for idx in elite_indices:
                    counts[sequences[idx][t]] += 1
                dist[t] = (counts + 0.01 * priors) / (counts.sum() + 0.01 * priors.sum())
        if best_sequence is None:
            return available_actions[0][0]
        best_action_idx = best_sequence[0]
        return available_actions[best_action_idx][0]

    def _evaluate_sequence(self, h, z, panel_idx, action_indices, available_actions) -> float:
        """
        评估一个动作序列的价值
        Keyword arguments:
        h -- 初始隐藏状态
        z -- 初始潜在变量
        panel_idx -- 初始面板索引
        action_indices -- 动作索引序列
        available_actions -- 可用动作列表
        Return arguments:
        float -- 序列的总奖励（包含惩罚项）
        """
        device = h.device
        # 当前 imagined rollout 的状态
        h_cur = h
        z_cur = z
        if isinstance(panel_idx, torch.Tensor):
            p_cur_idx = int(panel_idx.item())
        else:
            p_cur_idx = int(panel_idx)
        total_reward = 0.0
        discount = 1.0
        # 用于循环/卡住检测
        semantic_history = []
        panel_sequence = [p_cur_idx]
        prev_pred_panel = p_cur_idx
        stuck_steps = 0
        with torch.no_grad():
            for t, a_sel in enumerate(action_indices):
                # 解析动作
                action_str, verb_idx, target_idx, _atype = available_actions[a_sel]
                verb_t = torch.tensor([verb_idx], device=device)
                target_t = torch.tensor([target_idx], device=device)
                p_t = torch.tensor([p_cur_idx], device=device)
                h_next, z_next, r_pred, d_pred, p_logits = self.model.imagine_step(
                    h_cur.unsqueeze(0),
                    z_cur.unsqueeze(0),
                    verb_t,
                    target_t,
                    p_t
                )
                h_cur = h_next.squeeze(0)
                z_cur = z_next.squeeze(0)
                p_next_idx = int(torch.argmax(p_logits, dim=-1).item())
                # 基础外在奖励
                extrinsic = float(r_pred.item())
                # 拓扑新颖度奖励（面板越少访问越高）
                p_next_id = self._panel_id_from_idx(p_next_idx)
                intrinsic = float(self.topo_map.get_intrinsic_reward(p_next_id))
                # 动作奖励（鼓励未尝试动作）
                p_cur_id = self._panel_id_from_idx(p_cur_idx)
                frontier_bonus = 0.0
                if p_cur_id in self.topo_map.frontier_actions:
                    if action_str in self.topo_map.frontier_actions[p_cur_id]:
                        frontier_bonus = 1.0
                # 已知连通性校验/惩罚（基于真实拓扑图）
                known_pen = 0.0
                if (p_cur_id, action_str) in self.topo_map.edges:
                    known_dsts = self.topo_map.edges[(p_cur_id, action_str)]
                    if p_cur_id in known_dsts:
                        known_pen += self.known_loop_penalty
                    imagined_seen_ids = {self._panel_id_from_idx(x) for x in panel_sequence}
                    if len(known_dsts.intersection(imagined_seen_ids)) > 0:
                        known_pen += 0.5 * self.known_loop_penalty
                # 语义循环惩罚
                state_action = (p_cur_idx, verb_idx)
                semantic_pen = 0.0
                if state_action in semantic_history:
                    semantic_pen += SEMANTIC_LOOP_PENALTY
                semantic_history.append(state_action)
                # 状态循环惩罚
                state_loop_pen = 0.0
                panel_sequence.append(p_next_idx)
                if len(panel_sequence) >= 3:
                    if panel_sequence[-1] == panel_sequence[-3] and panel_sequence[-1] != panel_sequence[-2]:
                        state_loop_pen += STATE_LOOP_PENALTY
                # 卡住惩罚（连续不换面板）
                stuck_pen = 0.0
                if p_next_idx == prev_pred_panel:
                    stuck_steps += 1
                else:
                    stuck_steps = 0
                prev_pred_panel = p_next_idx
                if stuck_steps >= 2:
                    stuck_pen = STUCK_PENALTY * float(stuck_steps)
                step_value = (
                    extrinsic
                    + self.intrinsic_weight * intrinsic
                    + self.frontier_weight * frontier_bonus
                    - known_pen
                    - semantic_pen
                    - state_loop_pen
                    - stuck_pen
                )
                total_reward += discount * step_value
                discount *= GAMMA
                p_cur_idx = p_next_idx
                if float(d_pred.item()) > 0.5:
                    break
        return float(total_reward)


# ==================== 训练函数 ====================

def train_world_model(model, buffer, optimizer, device):
    """
    训练世界模型
    Keyword arguments:
    model -- 世界模型实例
    buffer -- 经验回放缓冲区
    optimizer -- 优化器
    device -- 设备
    Return arguments:
    Dict -- 包含各项损失平均值的字典
    """
    if len(buffer) < BATCH_SIZE:
        return {}
    losses = defaultdict(list)
    for _ in range(TRAIN_EPOCHS):
        batch = buffer.sample(BATCH_SIZE)
        if batch is None:
            continue
        # 准备数据
        states = batch['states'].to(device)
        verbs = batch['verbs'].to(device)
        targets = batch['targets'].to(device)
        rewards = batch['rewards'].to(device)
        panels = batch['panels'].to(device)
        dones = batch['dones'].to(device)
        # 前向传播
        outputs = model.forward_seq(states, verbs, targets, panels)
        # 计算各项损失
        reward_loss = F.mse_loss(outputs['rewards'], rewards)
        done_loss = F.binary_cross_entropy(outputs['dones'], dones)
        panel_logits_flat = outputs['panel_logits'].reshape(-1, MAX_PANELS)
        panels_flat = panels.reshape(-1)
        panel_loss = F.cross_entropy(panel_logits_flat, panels_flat)
        recon_loss = F.mse_loss(outputs['state_recons'], states)
        # KL散度损失
        post_mean = outputs['post_means']
        post_logstd = outputs['post_logstds']
        prior_mean = outputs['prior_means']
        prior_logstd = outputs['prior_logstds']
        kl_val = 0.5 * ((post_mean - prior_mean.detach()).pow(2) / torch.exp(2 * prior_logstd.detach()) + torch.exp(2 * post_logstd - 2 * prior_logstd.detach()) - 2 * (post_logstd - prior_logstd.detach()) - 1).sum(dim=-1).mean()
        alpha = 0.8
        loss_prior = 0.5 * torch.sum((post_mean.detach() - prior_mean).pow(2) / torch.exp(2 * prior_logstd) + torch.exp(2 * post_logstd.detach() - 2 * prior_logstd) - 2 * (post_logstd.detach() - prior_logstd) - 1, dim=-1).mean()
        loss_posterior = 0.5 * torch.sum((post_mean - prior_mean.detach()).pow(2) / torch.exp(2 * prior_logstd.detach()) + torch.exp(2 * post_logstd - 2 * prior_logstd.detach()) - 2 * (post_logstd - prior_logstd.detach()) - 1, dim=-1).mean()
        kl = alpha * loss_prior + (1 - alpha) * loss_posterior
        # 总损失
        total_loss = (W_REWARD * reward_loss +
                     W_DONE * done_loss +
                     W_PANEL * panel_loss +
                     W_RECON * recon_loss +
                     W_KL * kl)
        # 反向传播和优化
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        # 记录损失
        losses['total'].append(total_loss.item())
        losses['reward'].append(reward_loss.item())
        losses['done'].append(done_loss.item())
        losses['panel'].append(panel_loss.item())
        losses['kl'].append(kl.item())
    return {k: sum(v)/len(v) for k, v in losses.items()}


def train():
    """
    主训练循环函数
    """
    global STATE_DIM, panel_counter, verb_counter, target_counter
    # 初始化环境
    env = GameEnv()
    init_state = env.reset()
    STATE_DIM = len(init_state)
    # 初始化模型
    device = torch.device("cuda")
    model = PanelConditionedWorldModel(STATE_DIM).to(device)
    # 加载已有模型
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    # 初始化优化器和规划器
    optimizer = optim.Adam(model.parameters(), lr=LR)
    topo_map = TopologicalMap()
    planner = StructuredCEMEnhancedTopologicalPlanner(model, topo_map)
    replay = SequenceReplayBuffer(REPLAY_CAPACITY, SEQ_LEN)
    total_steps = 0
    # 主训练循环
    for ep in range(1, NUM_EPISODES + 1):
        state = env.reset()
        done = False
        # 初始化模型状态
        h, z = model.initial_state(1, device)
        h, z = h.squeeze(0), z.squeeze(0)
        # 编码初始观测
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=device)
            obs_embed = model.encode(state_t)
            post_mean, post_logstd = model.posterior(h.unsqueeze(0), obs_embed.unsqueeze(0))
            z = model.sample_latent(post_mean, post_logstd).squeeze(0)
        episode_reward = 0.0
        episode_data = []
        # 回合循环
        for step in range(1, MAX_STEPS_PER_EPISODE + 1):
            # 获取可用动作
            avail = env.get_available_actions()
            while not avail:
                time.sleep(0.2)
                avail = env.get_available_actions()
            # 获取当前面板信息
            panel_id = env.get_panel_info()
            if panel_id not in panel_vocab and panel_counter < MAX_PANELS:
                panel_vocab[panel_id] = panel_counter
                panel_counter += 1
            panel_idx = panel_vocab.get(panel_id, 0)
            # 构建可用动作列表
            available_actions = []
            for a in avail:
                verb_idx, target_idx = get_action_indices(a)
                verb_str = list(verb_vocab.keys())[list(verb_vocab.values()).index(verb_idx)] if verb_idx in verb_vocab.values() else "unknown"
                atype = get_action_type(verb_str)
                available_actions.append((a, verb_idx, target_idx, atype))
            # 选择动作（预热期随机，之后使用规划）
            if total_steps < WARMUP_STEPS:
                action = random.choice(avail)
                verb_idx, target_idx = get_action_indices(action)
            else:
                action = planner.plan(h, z, panel_idx, available_actions)
                verb_idx, target_idx = get_action_indices(action)
            # 执行动作
            next_state, reward, done = env.step(action)
            next_panel_id = env.get_panel_info()
            topo_map.update(
                prev_panel=panel_id,
                action=action,
                current_panel=next_panel_id,
                available_actions=avail
            )
            intrinsic_r = topo_map.get_intrinsic_reward(next_panel_id)
            episode_reward += reward + 0.1 * intrinsic_r
            total_steps += 1
            # 记录数据
            episode_data.append({
                'state': state,
                'verb_idx': verb_idx,
                'target_idx': target_idx,
                'reward': reward,
                'panel_idx': panel_idx,
                'done': float(done)
            })
            # 更新模型状态
            with torch.no_grad():
                verb_t = torch.tensor([verb_idx], device=device)
                target_t = torch.tensor([target_idx], device=device)
                panel_t = torch.tensor([panel_idx], device=device)
                next_state_t = torch.tensor(next_state, dtype=torch.float32, device=device)
                # 想象一步
                h, z, _, _, _ = model.imagine_step(h.unsqueeze(0), z.unsqueeze(0), verb_t, target_t, panel_t)
                h, z = h.squeeze(0), z.squeeze(0)
                # 使用真实观测更新潜在变量
                obs_embed = model.encode(next_state_t)
                post_mean, post_logstd = model.posterior(h.unsqueeze(0), obs_embed.unsqueeze(0))
                z = model.sample_latent(post_mean, post_logstd).squeeze(0)
            state = next_state
            if done:
                break
        # 将回合数据加入回放缓冲区
        replay.push_episode(episode_data)
        # 训练模型
        if total_steps > WARMUP_STEPS and ep % 2 == 0:
            train_losses = train_world_model(model, replay, optimizer, device)
        # 定期保存模型和打印信息
        if ep % 10 == 0:
            torch.save(model.state_dict(), MODEL_PATH)
            mode = "WARMUP" if total_steps < WARMUP_STEPS else "PLAN"
            print(f"[Ep {ep}] {mode} steps={step} R={episode_reward:.2f} "
                  f"buf={len(replay)} vocab(v={verb_counter},t={target_counter},p={panel_counter})")
        # 每回合结束保存模型
        torch.save(model.state_dict(), MODEL_PATH)
    print("Done!")


if __name__ == "__main__":
    train()
