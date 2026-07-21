# 书籍背包功能开发计划

通过对整个源码项目的系统分析（包括 Core数据结构、Design常量及结算机制、UI交互面板、Event事件流等），现制定以下针对“书籍背包”功能的开发计划：

## 1. 数据结构层 (Data Structure)
**目标:** 在现有的角色属性结构和行为状态中增加书籍管理所需的数据承载点。
*   **修改文件:** `Script/Core/game_type.py`
    *   在 `Character` 类的 `__init__` 函数中，新增 `self.book_bag: Set[str] = set()`（以书本 UID 的集合形式记录玩家所拥有的书籍）。
    *   在 `Behavior` 类的 `__init__` 函数中，新增 `self.read_book_id: str = ""` 记录正在阅读的书籍 UUID，以及 `self.book_name: str = ""` 用于事件文本动态替换（对应 `{BookName}`）。

## 2. 常量定义层 (Constants)
**目标:** 注册相关功能所需的游戏常量，确保状态机和面板路由正确流转。
*   **修改文件:** `Script/Design/constant/panel.py`
    *   增加 `BOOK_BAG = 14` (或下一个可用递增值) 用于标识书籍背包UI面板。
*   **修改文件:** `Script/Design/constant/instruct.py`
    *   增加主动类指令 `READ_BOOK = 74`，供玩家主动唤起。
*   **修改文件:** `Script/Design/constant/behavior.py`
    *   增加角色行为常量 `READ_BOOK = 74`。
*   **修改文件:** `Script/Design/constant/premise.py`
    *   增加前提条件常量 `HAVE_BOOK = "have_book"`，在判断指令是否可见时检查背包中是否有书。
*   **修改文件:** `Script/Design/constant/behavior_effect.py`
    *   增加行为结算效应 `READ_BOOK = "read_book"`。

## 3. UI与交互层 (UI & Flow)
**目标:** 增加书籍背包的查看与操作界面。
*   **新增文件:** `Script/UI/Panel/book_bag_panel.py`
    *   参考现存的 `food_bag_panel.py`，编写 `BookBagPanel` 面板系统。
    *   实现 `draw()` 方法：利用 `PageHandlePanel` 渲染 `self.book_bag` 内拥有的书籍列表。
    *   实现书籍详情展示：利用已有的 `game_config.config_book[uid]` 获取 `info` 并通过富文本渲染。
    *   实现 `read_book()` 阅读操作接口：
        *   将当前角色的 `behavior.behavior_id` 设为 `constant.Behavior.READ_BOOK`。
        *   绑定 `behavior.read_book_id` 及 `behavior.book_name`。
        *   配置阅读时间（例如 `behavior.duration = 10` 等，或者依据书籍体量设定）。
        *   调用 `update.game_update_flow(duration)` 进行时间步进。
        *   结束后重置面板并回到主场景：`cache.now_panel_id = constant.Panel.IN_SCENE`。
*   **新增文件:** `Script/UI/Flow/book_bag_flow.py`
    *   注入面板控制流：使用 `@handle_panel.add_panel(constant.Panel.BOOK_BAG)` 装饰器引导界面打开 `BookBagPanel`。

## 4. 指令与事件绑定 (Instruct & Event)
**目标:** 允许玩家从主场景下达“阅读”或“查看书籍”指令，以及将结算行为与事件广播文本接轨。
*   **修改文件:** `Script/Design/instruct/active.py`
    *   新建 `@handle_instruct.add_instruct(constant.Instruct.READ_BOOK, ...)` 装饰器节点。
    *   玩家触发指令时将面板切换至：`cache.now_panel_id = constant.Panel.BOOK_BAG`。
*   **修改数据:** `data/event/default.json`
    *   添加阅读行为配套事件，绑定 `"status_id": "74"` (与 Behavior 一致)，并将 `"settle": {"read_book": 1}` 加入事件结算对象中。
    *   将事件的输出文本配置为类似 `"{Name}翻开了{BookName}，安静地阅读了一会儿。"`。
*   **修改文件:** `Script/UI/Panel/draw_event_text_panel.py`
    *   在解析事件格式化占位符的地方增加 `BookName=character_data.behavior.book_name`，从而正确输出被读书籍名。

## 5. 属性效果结算逻辑 (Settle Logic)
**目标:** 当事件流程完成阅读后，正确赋予对应的学科或经验奖励。
*   **修改文件:** `Script/Settle/character_behavior.py` (或专门结算文件)
    *   新建 `handle_read_book` 方法并通过 `@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.READ_BOOK)` 注册。
    *   在实现内取出缓存的正在阅读的 `read_book_id`。
    *   获取对象书籍（即 `game_config.config_book[read_book_id]`），遍历其 `settle_list`。
    *   触发书籍中携带的所有增益函数（例如对 `add_small_chinese_experience` 的直接调用），从而精准执行 `default.json` 书籍数据文件里已配好的经验收益项。