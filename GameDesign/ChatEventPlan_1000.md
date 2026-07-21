# 《Dieloli》1000种闲聊事件（状态ID：5）扩充策划案

## 1. 设计目标与概述
在《Dieloli》的事件系统中，“闲聊（status_id: 5）”是触发频率最高、最能体现角色性格差异与社交关系深度的日常行为。
为了避免文本重复带来的机械感，提升游戏的生活流沉浸体验，本策划案旨在通过**“维度正交法”**，规划1000种互不重复、语境各异的闲聊事件。
规划过程不罗列具体事件文本，而是通过提取【性格】、【社交关系】、【场景/时间】、【技能/话题】以及【阶段（前摇/结算）】五大前提维度（Premise），建立系统性的生成矩阵与规划框架。

---

## 2. 维度拆解与标签池（严格对应 default.json 键名）

为实现1000种变化的排列组合，我们首先规范化将参与计算的底层前提（Premise）与结算（Settle）标签，**这些标签必须与 `default.json` 中的实际键名完全一致**：

### 2.1 性格与心理维度（Personality & Psychology）
*   **外向型：** 活泼 (`is_lively`), 热情 (`is_enthusiasm`), 自信 (`is_self_confidence`), 幽默 (`is_humor_man`), 直率 (`is_staraightforward`), 敏锐 (`is_keen`)
*   **内向型：** 孤僻 (`is_solitary`), 自卑 (`is_inferiority`), 低调 (`is_low_key`), 悲观 (`is_pessimism`)
*   **极端/特殊型：** 放纵 (`is_indulge`), 沉重/病娇 (`is_heavy_feeling`), 傲慢 (`arrogant_is_height`), 幼稚 (`is_childish`), 严谨 (`is_rigorous`), 自私 (`is_selfish`), 浮躁 (`is_impetuous`)
*   **性别与身份：** 女性 (`is_woman`, `target_is_woman`), 扶她或女性 (`is_futa_or_woman`), 学生 (`is_student`, `target_is_student`)

### 2.2 社交关系维度（Social Relationship）
*   **陌生/一般：** 仅有目标 (`have_target`), 目标存活 (`target_is_live`), 目标不是陌生人 (`target_not_stranger`)
*   **友好：** 目标宽容 (`target_is_tolerant`), 目标活泼 (`target_is_lively`), 目标幼稚 (`target_is_childish`)
*   **暧昧/恋爱：** 超越友谊 (`target_is_beyond_friendship`), 自身超越友谊 (`is_beyond_friendship_target`), 倾慕 (`target_is_admire`), 爱慕 (`target_is_adore`)
*   **敌对/排斥：** 目标没有超越友谊 (`no_beyond_friendship_target`), 目标厌恶度高 (`target_antipathy_is_height`)

### 2.3 场景与环境维度（Scene & Environment）
*   **常规地点：** 教室 (`in_classroom`), 树林 (`in_grove`), 餐厅 (`in_restaurant`), 天台 (`in_rooftop_scene`), 喷泉 (`in_fountain`)
*   **时间/天气：** 满月 (`tonight_is_full_moon`), 睡眠时间 (`in_sleep_time`), 晴天 (`is_sunny`)
*   **特殊状态：** 体力低/疲惫 (`mp_is_low`, `is_tired`), 饥饿 (`hunger`), 裸体 (`is_naked`, `target_is_naked`), 场景内无其他目标 (`no_have_other_target_in_scene`)

### 2.4 话题与技能维度（Topics & Skills）
*   **文科/艺术：** 文学 (`literature_is_height`), 绘画 (`painting_skills_is_height`), 诗歌 (`poetry_is_height`), 艺术 (`art_is_height`), 写作 (`write_skills_is_height`)
*   **理科/神秘学：** 天文学低 (`astronomy_skills_is_low`), 占星学低 (`astrology_skills_is_low`), 宗教信仰低 (`religion_is_low`)
*   **表达能力：** 口才高 (`eloquence_skills_is_height`) vs 口才低 (`eloquence_skills_is_low`)
*   **成人/私密：** 自身性欲高 (`lust_is_hight` 注：严格保留此拼写), 性技巧高 (`sexual_skills_is_height`), 目标性欲高 (`target_lust_is_hight`), 目标性欲低 (`target_lust_is_low`), 目标羞耻心高 (`target_shame_is_height`)

### 2.5 身体数据与体型差异维度（Physical Data & Stature）
*   **身高差异：** 目标高于角色 (`target_is_height`), 目标低于角色 (`target_height_low`), 目标接近平均身高 (`target_average_height_similar`), 目标低于平均身高 (`target_average_height_low`)
*   **体型/体重差异：** 目标偏胖 (`target_average_stature_height`), 自身偏瘦 (`is_average_stature_low`), 目标体型匀称 (`target_average_stature_similar`)
*   **胸围特征：** 自身非平胸 (`chest_is_not_cliff`), 目标平胸 (`target_chest_is_cliff`), 目标非平胸 (`target_chest_is_not_cliff`)

### 2.6 常用结算标签（Settle Keywords）
*   **让目标自身进入聊天状态：** `let_target_chat_self`
*   **基础消耗：** `sub_small_mana_point`
*   **经验增加：** `add_small_eloquence_experience`, `add_small_performance_experience`
*   **好感与偏好变更：** `target_add_small_favorability`, `target_add_medium_favorability`, `target_sub_small_favorability`, `add_like_preference`, `add_dislike_preference`
*   **情绪增减：** `add_small_happy`, `target_add_small_happy`, `target_add_small_shame`, `add_small_lust`, `target_add_small_lust`, `add_small_rage`, `target_add_small_rage`, `add_small_antipathy`, `target_add_medium_antipathy`, `add_small_yearn`, `add_small_pain`, `add_small_depressed`, `target_add_small_depressed`

---

## 3. 1000种闲聊事件矩阵规划

根据底层切片逻辑（`start: 1` 行为发起与画面感，`start: 0` 行为结算与反馈），我们将00个事件划分为七大主题模块进行乘数扩充，确保每种情境下角色的反应独一无二。

### 模块 A：日常校园社交与八卦（规划数量：0种）
**核心定位：** 最基础的校园生活感，不涉及高深技能或极端情感。
**生成公式：** [4种基础性格] × [3种场景(教室/餐厅/树林)] × [2种口才高/低] × [2种前摇/8种不同性格目标的反馈] = 192+种

*   **A1 - 主动发起闲聊 (start: 1) (约48种)**
    *   *活泼/热情*在*教室*分享见闻（口才高/低表现不同）。
    *   *孤僻/自卑*在*树林*试图搭话但支支吾吾。
    *   *幼稚/幽默*在*餐厅*边吃东西边讲冷笑话。
*   **A2 - 闲聊的结算与目标反馈 (start: 0) (约152种)**
    *   目标觉得有趣：增加好感、增加快乐（对应 `target_add_small_favorability`, `target_add_small_happy`, `add_small_happy`, `add_like_preference`）。
    *   目标觉得无聊或敷衍：无变化或微降好感（对应 `target_sub_small_favorability`, `add_dislike_preference`）。
    *   性格碰撞：*严谨*的目标无情纠正*幼稚*角色的常识错误，造成小幅尴尬（增加羞耻 `target_add_small_shame`）。

### 模块 B：学术探讨与才艺展示（规划数量：0种）
**核心定位：** 带有明确技能门槛（Premise中要求技能等级高）的深度对话。
**生成公式：** [5种专业技能(文学/美术/音乐/天文/理科)] × [3种性格(自信/严谨/热情)] × [2种社交关系(普通/崇拜)] × [5种演进相] = 0种

*   **B1 - 技能话题发起 (start: 1) (约60种)**
    *   *文学高*+*悲观*：聊起伤痛文学和悲剧宿命。
    *   *天文高*+*满月之夜*+*天台*：指着星空科普星座传说。
    *   *美术高*+*自信*：对着对方的长相品头论足，高谈阔论骨相美。
*   **B2 - 技能话题反馈 (start: 0) (约90种)**
    *   目标崇拜（`target_is_admire`）：听得入神，增加大量好感，自身增加傲慢（对应 `target_add_medium_favorability`, `add_small_arrogant`, `let_target_chat_self`）。
    *   对方技能同样高：产生学术共鸣，惺惺相惜（对应双向加快乐 `add_small_happy`, `target_add_small_happy`, 以及 `add_small_eloquence_experience`）。
    *   对方完全听不懂：产生无奈感（对应 `add_small_antipathy`, `target_add_small_depressed`, `add_dislike_preference`）。

### 模块 C：情感试探与暧昧互动（规划数量：0种）
**核心定位：** 推动游戏恋爱线发展的核心交互，含有大量微表情与肢体暗示。
**生成公式：** [4种暧昧阶段(倾慕/爱慕/超越友谊/单向暗恋)] × [4种性格(自卑/自信/沉重/活泼)] × [3种羞耻度高/低] × [5种衍生] = 240+种

*   **C1 - 暧昧的搭讪与试探 (start: 1) (约0种)**
    *   *自卑*+*超越友谊*+*高羞耻*：红着脸扯着对方衣角小声说话，眼神躲闪。
    *   *自信*+*爱慕*+*低羞耻*：极具侵略性的贴近对方耳边低语，吐气如兰。
    *   *沉重(病娇)*+*超越友谊*：通过闲聊疯狂查岗，字里行间询问对方今天接触了谁。
*   **C2 - 暧昧互动的反馈 (start: 0) (约0种)**
    *   目标羞涩回应：增加双向欲望，增加快乐与羁绊（对应 `target_add_small_lust`, `add_small_lust`, `target_add_small_happy`, `add_small_happy`, `target_add_small_favorability`, `add_like_preference`）。
    *   目标被撩拨得不知所措：目标增加羞耻度、小幅惊恐（对应 `target_add_small_shame`, `target_add_small_fear`）。
    *   自作多情的翻车：碰钉子，自身增加痛苦和抑郁（对应 `add_small_pain`, `add_small_depressed`, `target_sub_small_favorability`）。

### 模块 D：成人向与私密性骚扰话题（规划数量：0种）
**核心定位：** 符合游戏成人标签，基于高性欲与高性技巧展开的越界对话与言语调戏。
**生成公式：** [3种欲望度(高/中/低)] × [2种状态(裸体/穿着)] × [5种目标反应] × [5种细分场景] = 0种

*   **D1 - 越界话题发起 (start: 1) (约0种)**
    *   *高性欲*+*放纵*+*低羞耻*：光天化日下讲述荤段子或直接调戏对方的敏感部位。
    *   *高性欲*+*自卑*+*高羞耻*：结结巴巴、面红耳赤地询问两性方面的私密知识。
    *   *裸体状态下*+*放纵*：毫不在意走光，故意用色情话题挑逗对方的底线。
*   **D2 - 越界话题反馈 (start: 0) (约0种)**
    *   目标迎合（目标同样高性欲/放纵）：两人越聊越火热，眼神拉丝（对应 `target_add_medium_lust`, `add_medium_lust`, `target_add_medium_favorability`, `let_target_chat_self`）。
    *   目标反感/辱骂（目标低性欲/高严谨）：一把推开或大骂结束话题（对应双向厌恶 `add_small_antipathy`, `target_add_medium_antipathy`, `target_add_small_rage`, `add_small_shame`, `add_small_pain`）。
    *   目标被成功调教（目标羞耻度高但好感度极高）：半推半就地红着脸听完（对应 `target_add_small_shame`, `target_add_small_lust`）。

### 模块 E：负面情绪与冲突交锋（规划数量：0种）
**核心定位：** 嫉妒、傲慢、反感驱动下的“阴阳怪气”与无效沟通。
**生成公式：** [4种负面原罪(嫉妒/傲慢/暴怒/厌恶)] × [3种性格(低调/活泼/悲观)] × [8种反馈组合] = 96+种

*   **E1 - 恶意的闲聊 (start: 1) (约40种)**
    *   *高嫉妒*+*低调*：用极其酸涩和阴阳怪气的语气开启话题，暗自吃醋。
    *   *高傲慢*+*自信*：居高临下地对对方进行说教和贬低式闲聊（打压式沟通）。
    *   *高厌恶*（对目标）：虽然在闲聊，但满脸不耐烦，四处张望就是不看对方。
*   **E2 - 冲突结算 (start: 0) (约60种)**
    *   目标同样暴躁：闲聊升级为言语冲突，不欢而散（对应双向减好感 `target_sub_small_favorability`, 双向愤怒 `add_small_rage`, `target_add_small_rage`, 痛苦 `add_small_pain`）。
    *   目标软弱/自卑：被贬低到默默掉眼泪（对应目标抑郁恐惧 `target_add_small_depressed`, `target_add_small_fear`, 发起方傲慢快乐 `add_small_arrogant`, `add_small_happy`）。

### 模块 F：极端环境与特殊生理状态（规划数量：0种）
**核心定位：** 由于角色自身的极端生理状态或特殊天气（系统随机生成）引发的特殊情境对话。
**生成公式：** [5种状态(极度疲惫/饥饿/深夜/下雨/孤男寡女)] × [3种应对性格] × [10种互动反馈] = 0种

*   **F1 - 状态驱动发起 (start: 1) (约60种)**
    *   *极度疲惫* (mp_is_low)：气喘吁吁、有气无力地搭话，眼皮打架随时要睡着。
    *   *饥饿* (hunger)：话题三句不离吃的，甚至盯着对方的脖子咽口水（幽默/病娇双判定）。
    *   *深夜*+*天台*：借着夜风，抛下白天的伪装，开启关于人生和遗憾的私密闲聊。
*   **F2 - 状态结算 (start: 0) (约90种)**
    *   聊着聊着睡着了（对应 `sub_small_mana_point`，不额外增加其它情绪）。
    *   聊着聊着肚子叫了（引发尴尬增加自身羞耻 `add_small_shame`；若目标宽容则觉得可爱 `target_add_small_favorability`）。
    *   深夜天台的交心（大幅增加羁绊，清空双方负面情绪，对应 `target_add_medium_favorability`, `sub_large_pain`, `sub_large_depressed`, `target_sub_large_pain`）。

### 模块 G：身体形态与发育比较（规划数量：0种）
**核心定位：** 基于双方的身高、体型、胸围等外貌与身体发育特征产生的比较、调侃、自卑或羡慕的闲聊对话。
**生成公式：** [3种差异特征(身高/体型/胸围)] × [4种性格(自卑/自信/活泼/傲慢)] × [4种关系或情绪反应] × [2种状态(发起/结算)] = 96+种

*   **G1 - 身体差异的话题发起 (start: 1) (约40种)**
    *   *身高差异* (`target_is_height`) + *活泼*：开玩笑般地比划自己与对方的身高，甚至想垫脚摸对方的头。
    *   *胸围差异* (`target_chest_is_not_cliff`) + *自卑/嫉妒*：视线不由自主地停留在对方饱满的胸部，小声感叹或低头看自己的脚尖。
    *   *体型差异* (`target_average_stature_height`) + *傲慢*：居高临下地调侃对方最近是不是胖了，带有一丝得意的炫耀。
*   **G2 - 身体比较的结算与反馈 (start: 0) (约60种)**
    *   目标大方回应/炫耀：如果被夸奖身材，目标增加快乐和傲慢（对应 `target_add_small_happy`, `target_add_small_arrogant`）。
    *   目标感到难堪/羞耻：如果被指出平胸 (`target_chest_is_cliff`) 或偏胖，目标增加羞耻或愤怒（对应 `target_add_small_shame`, `target_add_small_rage`）。
    *   温柔的安慰：如果发起方因身体数据自卑（如 `is_average_stature_low`），对方摸摸发起方的头，表示现在这样就很可爱（大幅增加好感 `target_add_medium_favorability`，降低痛苦 `sub_small_pain`）。

---

## 4. 关键变量与前提（Premise）组合一览表示例

为了在实际录入 `default.json` 且使用大语言模型撰写文本时确保绝对的正交与不重复，须严格遵守以下变量对照池：

| 维度类别 | 标签变量 (Premise Key) | 在闲聊中的语境要求与限制（供AI生成参考） |
| :--- | :--- | :--- |
| **表达能力** | `eloquence_skills_is_height` | 描述必须体现：滔滔不绝、引人入胜、引经据典、节奏把控极佳。 |
| **表达能力** | `eloquence_skills_is_low` | 描述必须体现：结结巴巴、词不达意、容易冷场、气氛尴尬。 |
| **性格态度** | `is_rigorous` | 描述必须体现：严肃、爱纠错、一板一眼、甚至有些不解风情。 |
| **性格态度** | `is_indulge` | 描述必须体现：毫无顾忌、口无遮拦、姿态慵懒、缺乏边界感。 |
| **极端情绪** | `is_heavy_feeling` | 描述必须体现：充满占有欲、试探性极强、眼神拉丝或死死盯住对方。 |
| **场景时间** | `tonight_is_full_moon` | 文本质感必须偏向：唯美、神秘、受到月光影响的情感外露。 |
| **生理状态** | `mp_is_low` | 描写必须包含：语速缓慢、声音细若蚊蝇、动作迟缓或呼吸沉重。 |
| **私密倾向** | `lust_is_hight` | 描写必须包含：眼神聚焦于敏感部位、话题带颜色、呼吸急促等暗示。 |

---

## 5. 落地执行与管线建议
1.  **AI批量生成提示词（Prompt）约束：** 
    *   在调用大语言模型批量生成这00条文本时，务必将上述模块化拆解（如“生成模块C2，前提是：自卑+超越友谊+高羞耻的反馈”）转化为精确的Prompt。
    *   严格应用《事件系统设计策划案》中的“不超过5句话”、“不直接描写对话（不使用双引号）”、“避免超自然元素”这三大铁律。
    *   **严格遵循 JSON 格式输出**：必须使用 `default.json` 要求的字段结构，包含 `uid`, `adv_id`, `status_id: "5"`, `start`, `text` (使用宏如 `{Name}` 和 `{TargetName}`), `premise` 对象, 以及 `settle` 对象。
    *   **键名绝对匹配**：绝不捏造变量，只能使用第 2 节列出的合法键名（如使用 `lust_is_hight` 而不是 `lust_is_high`，使用 `let_target_chat_self` 让目标互动等）。
2.  **触发权重分层控制：** 带有极端性格（如沉重、放纵）或极端状态（裸体、满月）的闲聊事件，由于其前提极度严苛，自带低频触发属性；而模块A的日常寒暄前提较少，极易产生“刷屏”效应。建议在游戏底层逻辑中，为匹配了更多 Premise 标签的事件赋予**更高优先级的权重**，确保玩家千辛万苦凑齐特殊前置条件后，那些特殊的闲聊事件能够绝对优先地展现出来。

---

## 6. 创作进度记录
* **真实进度统计口径：** 以后以“严格带齐生成公式维度”的事件作为真实完成进度；单纯写入 `ChatEvents_Generated.json` 但缺少公式维度标签的事件，只计入“实际事件数”，不计入真实完成进度。
* **当前真实进度：** 808/1000
* **当前实际事件数：** 1234/1000
* **输出文件：** `GameDesign/ChatEvents_Generated.json`
* **统计时间：** 2026-06-06
* **归类规则：** `ChatEvents_Generated.json` 没有显式模块字段，按 `premise`/`settle` 键回推模块；同一事件可匹配多个模块时，按 `G > D > C > F > B > E > A` 优先级只归入一个模块，避免重复计数。
* **真实进度表：**

| 模块 | 生成公式 | 规划公式槽 | 实际事件数 | 严格带齐公式维度事件 | 唯一公式槽覆盖 |
| :--- | :--- | ---: | ---: | ---: | ---: |
| A 日常校园社交与八卦 | 4种基础性格 × 3种场景 × 2种口才 × start差分 | 48 | 148 | 40 | 17 |
| B 学术探讨与才艺展示 | 5种专业技能 × 3种性格 × 2种社交关系 × 5种演进相 | 150 | 193 | 142 | 124 |
| C 情感试探与暧昧互动 | 4种暧昧阶段 × 4种性格 × 3种羞耻度 × 5种衍生 | 240 | 269 | 166 | 163 |
| D 成人向与私密性骚扰话题 | 3种欲望度 × 2种状态 × 5种目标反应 × 5种细分场景 | 150 | 97 | 92 | 14 |
| E 负面情绪与冲突交锋 | 4种负面原罪 × 3种性格 × 8种反馈组合 | 96 | 156 | 101 | 96 |
| F 极端环境与特殊生理状态 | 5种状态 × 3种应对性格 × 10种互动反馈 | 150 | 204 | 140 | 126 |
| G 身体形态与发育比较 | 3种差异特征 × 4种性格 × 4种关系或情绪反应 × 2种状态 | 96 | 167 | 127 | 96 |

* **公式差分完成情况：**
  * 模块 A：严格公式事件 40 条，唯一公式槽 17/48。主要缺口是口才维度缺失，`start: 0` 结算反馈不足。
  * 模块 B：严格公式事件 142 条，唯一公式槽 124/150。近期补入 `painting_skills_is_height` + `is_self_confidence` + `target_is_admire` + `start: 0` 的绘画高熟练自信性格崇拜关系下中性延续反馈槽（`sub_small_mana_point` + `add_small_eloquence_experience` + `let_target_chat_self`）。主要缺口仍是普通/崇拜关系下的其他 `start: 0` 技能反馈和不同演进相；后续优先补 B 的低覆盖 `start: 0` 反馈槽，再补 F/C 的缺失公式维度标签。
  * 模块 C：严格公式事件 166 条，唯一公式槽 163/240。本次补入 `target_is_admire` + `is_self_confidence` + `target_shame_is_height` + `start: 0` 的倾慕关系下自信性格面对目标高羞耻的退缩压抑反馈槽（`sub_small_mana_point` + `add_small_eloquence_experience` + `target_add_small_depressed` + `target_sub_small_favorability` + `add_dislike_preference` + `let_target_chat_self`）。主要缺口仍是高/低羞耻和目标性欲维度下，不同性格维度的其他 `start: 0` 结算反馈。
  * 模块 D：严格公式事件 92 条，唯一公式槽 14/150。事件维度较完整，但目标反应和场景组合重复过多；按本轮优先级暂不补 D。
  * 模块 E：严格公式事件 101 条，唯一公式槽 96/96。近期已补入 `arrogant_is_height` + `is_low_key` + `target_is_pessimism` + `start: 0` 的傲慢低调后目标悲观退让反馈槽（`sub_small_mana_point` + `add_small_eloquence_experience` + `add_small_arrogant` + `target_add_small_depressed` + `target_sub_small_favorability` + `add_dislike_preference` + `let_target_chat_self`）。E 模块规划公式槽已补齐，后续优先回到 C、B、F 的低覆盖槽。
  * 模块 F：严格公式事件 140 条，唯一公式槽 126/150。本次补入 `is_tired` + `is_low_key` + `target_is_lively` + `start: 0` 的极度疲惫下低调应对后目标活泼接话反馈槽（`sub_small_mana_point` + `add_small_eloquence_experience` + `target_add_small_favorability` + `target_add_small_happy` + `add_like_preference` + `let_target_chat_self`）。主要缺口仍是 `start: 0` 互动反馈，尤其是睡眠时间、孤立场景、疲惫状态和低体力状态下的其他目标反应组合；后续优先补 B 的低覆盖 `start: 0` 反馈槽，再补 F/C 的缺失公式维度标签。
  * 模块 G：严格公式事件 127 条，唯一公式槽 96/96。近期已补入 `target_average_stature_height` + `is_lively` + `target_is_admire` + `start: 0` 的体型差异活泼后目标倾慕回应反馈槽（`sub_small_mana_point` + `add_small_eloquence_experience` + `target_add_small_happy` + `target_add_small_arrogant` + `target_add_small_favorability` + `add_like_preference` + `let_target_chat_self`）。G 模块规划公式槽已补齐，后续优先回到 C、B、F 的低覆盖槽。
