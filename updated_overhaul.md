# 社交交互系统重构策划案 (Social Interaction System Overhaul)

## 1. 现状痛点分析 (Current Pain Points)
目前的交互机制本质上是“单体状态机的并联”，而非“系统的串联”，这导致了以下几个核心问题：

*   **单向行为与决策孤立**：角色 A 的 AI 目标搜索完全不考虑角色 B 的实时社交意图。当 A 决定和 B 说话时，A 会单方面进入交互状态，而 B 毫无察觉。
*   **状态强制篡改与逻辑断裂**：现有的交互通常通过底层逻辑（如 `settle_behavior_effect`）强行修改目标的 `state`。如果 B 正在做某项高优先级的专注任务（如吃饭、睡觉），会被瞬间强行打断并转头聊天，缺乏自然平滑的过渡和合理性判定。
*   **交互短视（一锤子买卖）**：交互双方仅通过 `target_character_id` 简单链接，一回合结算（加减属性、输出一段文本）后即告结束。无法承载复杂的、多阶段的互动逻辑（如：寒暄 -> 深入探讨 -> 告别），也无法支持需要多轮博弈的玩法（如下棋、打牌、争吵）。
*   **群体社交空白**：目前的架构强绑定于 1v1 目标，难以实现 3 人及以上的自然互动（如：多人聚餐、围观讨论、课堂教学）。

## 2. 核心构想：会话与场域驱动 (Session & Field Driven)
将交互从“指令驱动”转变为“会话握手”与“场域驱动”。

### 2.1 交互会话 (Interaction Session)
“交互会话”是角色之间建立的**契约**。任何涉及双人或多人的持续性互动，都必须建立在 Session 之上。
*   **状态共享**：Session 统一管理当前交互处于哪个阶段。
*   **数据隔离**：交互产生的临时变量（当前话题、谁该说话、游戏比分）均存放在 Session 中，而非角色本体。

### 2.2 社交场 (Social Field)
“社交场”是 Session 在地理空间（Scene）上的投影。
*   **可见性与氛围**：场域对外散发信息（如“A和B正在热烈讨论”），具有氛围值（尴尬、热烈、悲伤）。
*   **开放性**：定义其他人是否可以旁听、是否可以随时加入（如开放的牌局 vs 私密的告白）。

### 2.3 握手与响应协议 (Handshake Protocol)
重构行为选择逻辑，禁止单方面强制拉取，改为“请求-评估-响应”机制：
1.  **社交请求 (Request)**：角色 A 决定发起交互，创建会话并广播给目标 B（推入目标的信箱）。A 进入社交互动状态。
2.  **权重竞争 (Evaluate)**：目标 B 在下一次 AI Tick 时，除了评估自己的日常行为树（如扫地、吃饭），还会评估接收到的请求。
    *   *评估权重受以下因素影响：B对A的好感度、B当前的疲劳度、B当前行为的重要性（睡觉 > 扫地）、双方性格匹配度等。*
3.  **最终响应 (Response)**：
    *   **接受 (Accept)**：如果 B 评估后认为“响应请求”的权重最高，B 会中断当前行为，同步切换至 `交互中(STATUS_SOCIAL_INTERACTING)` 状态。
    *   **拒绝 (Reject)**：如果 B 当前行为权重更高（例如极度困倦或正在重要工作），则向 A 发送拒绝。A 退出等待状态，重新规划行为。

## 3. 技术框架实现现状与思路

目前游戏已初步接入此框架，实际数据结构及实现思路如下：

### 3.1 数据结构设计 (`game_type.py`)
目前已在 `game_type.py` 中实现了核心的 `InteractionSession` 类，并将其整合在 `Cache` 中：

```python
class InteractionSession:
    def __init__(self, initiator_id: int, target_ids: List[int], session_type: int):
        self.uid: str = ""                # 会话唯一id
        self.initiator: int = initiator_id# 发起者id
        self.members: List[int] = [initiator_id] + target_ids # 参与者id列表
        self.type: int = session_type     # 会话类型
        self.data: dict = {}              # 存放交互过程中的临时数据 (stage, atmosphere等)
        self.start_time: int = 0          # 开始时间戳
```
角色本体 (`Character`) 中新增了：
*   `social_requests`: 用于接收握手请求的列表（字典结构，包含发起者、会话UID、类型和权重）。
*   `active_session`: 当前正在参与的 Session UID。
*   `STATUS_SOCIAL_INTERACTING`: 角色处于会话中的专属状态。

### 3.2 会话管理与状态机
全局的 Session 管理由 `character_behavior.py` 中的 `update_social_sessions()` 接管，目前支持：
1.  **多阶段演进**：通过 `session.data['stage']` 随着时间推移演进会话阶段。
2.  **氛围波动**：随着时间自动计算并存储氛围值 `atmosphere`。
3.  **合法性校验与回收**：当参与者状态改变或不在同一场景时，Session 自动回收并清理成员状态。

### 3.3 玩家界面的社交场呈现
在 `in_scene_panel.py` 中已经实现了“社交场”的可视化。玩家进入场景时，可以直观地看到场景内的“互动群组”，并且系统允许玩家直接调用 `join_session` **作为第三人加入正在进行中的 NPC 闲聊**。

## 4. 后续开发计划与里程碑 (Roadmap)

### 阶段一：底层基础设施建设 (Foundation) 【已完成】
*   [x] 在 `game_type.py` 中定义 `InteractionSession` 核心数据结构。
*   [x] 在 `character_behavior.py` 中实现全局 Session 更新逻辑（超时演进、销毁、回收机制）。
*   [x] 在角色的属性中增加用于接收请求的信箱（`social_requests`）以及当前所处的 `active_session`。
*   [x] 新增状态 `STATUS_SOCIAL_INTERACTING`。

### 阶段二：握手协议与 AI 整合 (AI Handshake Protocol) 【进行中】
*   [x] 修改角色发起的交互（如 `character_chat_rand_character`），改为生成 Session 并发送至目标 `social_requests`。
*   [x] 修改角色行为树逻辑（`character_behavior.py`），新增优先处理入站请求的逻辑。
*   [ ] **待完善：引入真实的权重竞争（目前收到请求无条件接受），对比现有行为与社交请求的权重。**
*   [ ] **待完善：实现真正的拒绝反馈闭环（NPC 拒绝时发起方退出会话并重新规划行为，目前没有处理拒绝）。**

### 阶段三：会话处理器与重构基础交互 (Refactor Core Interactions) 【待办】
*   [ ] 设计 `BaseSessionHandler` 基础接口类，将 `update_social_sessions` 中的硬编码演进逻辑解耦出去。
*   [ ] 挑选最基础的“闲聊 (Chat)”功能，编写 `ChatSessionHandler`，实现双向 Session 内推进与属性结算。
*   [ ] 彻底清理历史遗留的 `handle_let_target_chat_self` 强行拉取状态的底层逻辑。

### 阶段四：群体社交与玩家接入 (Group & Player Integration) 【已初步完成】
*   [x] 扩展框架，实现 3 人及以上的 `InteractionSession` 支持。
*   [x] 开放“社交场”在场景地图中的注册 (`scene.social_fields`)，完善 UI 层面可视化展示。
*   [x] 允许玩家主动加入已存在的 Session (`join_session`)。
*   [ ] 修改玩家主动发起交互的指令入口，适配握手协议（目前玩家交互依然走老的一回合结算逻辑）。
*   [ ] 补充玩家被 NPC 拒绝时的 UI 提示与情感反馈机制。

### 阶段五：深度内容填充 (Content Expansion) 【待办】
*   [ ] 基于新的 Session 框架，开发复杂互动玩法：如多人聚餐、课堂教学（接管原有的让学生强行听课的逻辑）、多人争执、棋牌游戏、会议等。
