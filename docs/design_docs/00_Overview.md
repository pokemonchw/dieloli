# DieLoli 整体架构与设计哲学概述

## 1. 设计哲学
DieLoli 的核心设计理念是**“底层沙盒，自由演算”**。
* **没有固定主线**：游戏类似于《环世界》(RimWorld) 和《矮人要塞》(Dwarf Fortress)。游戏世界由时间流逝和 NPC 的自主行为驱动，玩家只是其中的一个实体。
* **数据驱动一切**：游戏的大量业务逻辑（例如一个行为的触发条件是什么，这个行为会带来什么属性变化，行为完成后的下一阶段是什么）全部剥离到外部的 CSV 文件中，由策划工具编译为代码。
* **强解耦的引擎设计**：UI 渲染系统、核心框架流转系统、游戏业务逻辑三者被严格隔离。

## 2. 系统层次架构
整个系统从下至上可以分为以下四个层级：

### 2.1 UI 与引擎底层 (Engine & UI Layer - `Script/Core`)
摒弃了原生终端，采用 `PySide6` 自绘富文本界面 (`main_frame.py`)。它提供了 `window` 和输入队列机制。核心模块 `flow_handle` 不断轮询输入队列，派发指令给对应的面板或流程。

### 2.2 核心业务逻辑系统 (Game Logic Layer - `Script/Design`)
负责具体的业务流转。主要包括：
- **时间与地图模块 (`game_time.py`, `map_handle.py`)**：负责游戏内时间步的推进、季节变换、场景节点的寻路和地图资源管理。
- **角色实体模块 (`character.py`, `character_handle.py`)**：负责所有角色（包括数千个 NPC 和玩家）的数据载体、状态计算。
- **行为与结算系统 (`character_behavior.py`, `settle_behavior.py`)**：定义行为是如何产生和完成计算的。

### 2.3 数据与决策系统 (Data & Decision Layer - `StateMachine`, `Premise`, `Settle`, `tools/`)
这是 DieLoli 区别于一般 AVG 游戏的核心。
- 所有的行为前提 (Premise)、状态流转 (StateMachine)、结算结果 (Settle) 都通过 `tools/` 下的构建脚本将 `.csv` 文件编译成原生的 `.py` 代码，极大提升了每帧的演算速度。

### 2.4 AI 与大模型支持 (AI & LLM Integration Layer - `ai_api.py`, `event.py`)
- **生成式文本**：游戏事件中使用了大模型（Ollama 等）基于上下文（人物、动作、地点、前置条件）动态生成小说级文本。
- **强化学习框架**：提供 HTTP 接口，外部可将整个游戏作为强化学习的环境（RL Env）进行训练。

## 3. 游戏核心循环 (Game Main Loop)
1. 玩家/UI 发生交互（点击或输入）。
2. `Core/io_init.py` 捕获事件，压入队列。
3. 当前的交互流程 (`UI/Flow`) 被唤醒，处理输入。
4. 如果输入是推进时间的指令：
   - 遍历全图所有 NPC。
   - 调用 NPC 的状态机进行状态流转。
   - 结算状态改变，刷新角色属性。
   - 触发可能发生的全局或局部事件（此时可能调用 LLM 生成描述文本）。
   - 推进游戏内时间。
5. 更新 UI 面板显示。