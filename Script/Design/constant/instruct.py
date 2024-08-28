class Instruct:
    """指令id"""

    """
    =========================
    对话类指令
    =========================
    """
    CHAT = 0
    """ 闲聊 """
    ABUSE = 0
    """ 辱骂 """
    GENERAL_SPEECH = 1
    """ 演讲 """

    """
    =========================
    主动类指令
    =========================
    """
    EAT = 0
    """ 进食 """
    MOVE = 0
    """ 移动 """
    ESCAPE_FROM_CROWD = 0
    """ 逃离人群 """
    BUY_ITEM = 0
    """ 购买道具 """
    BUY_FOOD = 0
    """ 购买食物 """
    BUY_CLOTHING = 0
    """ 购买服装 """
    DRINK_SPRING = 0
    """ 喝泉水 """
    EMBRACE = 0
    """ 拥抱 """
    HAND_IN_HAND = 0
    """ 牵手 """
    LET_GO = 0
    """ 放手 """
    WEAR = 0
    """ 穿衣 """
    SELF_UNDRESS = 0
    """ 脱自己的衣服 """
    SEE_STAR = 0
    """ 看星星 """
    CLUB_ACTIVITY = 0
    """ 社团活动 """

    """
    =========================
    表演类指令
    =========================
    """
    SINGING = 0
    """ 唱歌 """
    DANCE = 0
    """ 跳舞 """
    PLAY_PIANO = 0
    """ 弹钢琴 """
    PLAY_GUITAR = 0
    """ 弹吉他 """

    """
    =========================
    猥亵类指令
    =========================
    """
    TOUCH_HEAD = 0
    """ 摸头 """
    KISS = 0
    """ 亲吻 """
    STROKE = 0
    """ 抚摸 """
    TOUCH_CHEST = 0
    """ 摸胸 """
    TARGET_UNDRESS = 0
    """ 脱下对方衣服 """
    STALKER = 0
    """ 尾行 """

    """
    =========================
    娱乐类指令
    =========================
    """
    PLAY_COMPUTER = 0
    """ 玩电脑 """
    DRAW = 0
    """ 画画 """
    RUN = 0
    """ 跑步 """

    """
    =========================
    战斗类指令
    =========================
    """

    """
    =========================
    学习类指令
    =========================
    """
    VIEW_THE_SCHOOL_TIMETABLE = 0
    """ 查看课程表 """
    ATTEND_CLASS = 0
    """ 上课 """
    TEACH_A_LESSON = 0
    """ 教课 """
    SELF_STUDY = 0
    """ 自习 """

    """
    =========================
    休息类指令
    =========================
    """
    REST = 0
    """ 休息 """
    SIESTA = 0
    """ 午睡 """
    SLEEP = 0
    """ 睡觉 """

    """
    =========================
    性爱类指令
    =========================
    """
    MASTURBATION = 0
    """ 手淫 """
    INVITE_SEX = 0
    """ 邀请做爱 """
    TOUCH_CLITORIS = 0
    """ 抚摸阴蒂 """
    TOUCH_PENIS = 0
    """ 抚摸阴茎 """
    TOUCH_ANUS = 0
    """ 抚摸肛门 """
    TOUCH_BUTT = 0
    """ 抚摸屁股 """
    TOUCH_BACK = 0
    """ 摸背 """
    PENIS_RUB_FACE = 0
    """ 阴茎蹭脸 """
    FINGER_INSERTION_VAGINAL = 0
    """ 手指插入阴道 """
    PLAY_CLITORIS = 0
    """ 玩弄阴蒂 """
    PLAY_PENIS = 0
    """ 玩弄阴茎 """
    FINGER_INSERTION_ANUS = 0
    """ 手指插入肛门 """
    PLAY_NIPPLE = 0
    """ 玩弄乳头 """
    LICK_VAGINAL = 0
    """ 舔阴 """
    LICK_PENIS = 0
    """ 舔阴茎 """
    LICK_ANUS = 0
    """ 舔肛门 """
    LICK_NIPPLE = 0
    """ 吸舔乳头 """
    LICK_FEET = 0
    """ 舔足 """
    LICK_BUTT = 0
    """ 舔屁股 """
    LICK_EARS = 0
    """ 舔耳朵 """
    LICK_BODY = 0
    """ 舔全身 """
    LICK_FACE = 0
    """ 舔脸 """
    TARGET_MOUTH_AND_HAND_SEX = 0
    """ 让对方手交口交 """
    TARGET_MOUTH_AND_CHEST_SEX = 0
    """ 让对方乳交口交 """
    TARGET_VACCUM_MOUTH_SEX = 0
    """ 让对方真空口交 """
    TARGET_DEEP_MOUTH_SEX = 0
    """ 让对方深喉 """
    MOUTH_AND_HAND_SEX = 0
    """ 给对方手交口交 """
    MOUTH_AND_CHEST_SEX = 0
    """ 给对方乳交口交 """
    VACCUM_MOUTH_SEX = 0
    """ 给对方真空口交 """
    DEEP_MOUTH_SEX = 0
    """ 给对方深喉 """
    SIX_NINE_SEX = 0
    """ 六九式 """
    INSERTION_VAGINAL = 0
    """ 插入阴道 """
    HITTING_UTERUS = 0
    """ 撞击子宫 """
    RIDING_INSERTION_VAGINAL = 0
    """ 骑乘位插入阴道 """
    BACK_INSERTION_VAGINAL = 0
    """ 背后位插入阴道 """
    BACK_RIDING_INSERTION_VAGINAL = 0
    """ 背后位骑乘插入阴道 """
    NO_PENIS_SEX = 0
    """ 磨豆腐 """
    INSERTION_ANUS = 0
    """ 肛交 """
    BACK_INSERTION_ANUS = 0
    """ 背后位肛交 """
    RIDING_INSERTION_ANUS = 0
    """ 骑乘位肛交 """
    BACK_RIDING_INSERTION_ANUS = 0
    """ 背后位骑乘肛交 """
    SEX_END = 0
    """ 结束性爱 """

    """
    =========================
    系统类指令
    =========================
    """
    SEE_ATTR = 0
    """ 查看属性 """
    SEE_OWNER_ATTR = 0
    """ 查看自身属性 """
    VIEW_CHARACTER_STATUS_LIST = 0
    """ 查看角色状态列表 """
    VIEW_CLUB_LIST = 0
    """ 查看社团列表 """
    VIEW_CLUB_INFO = 0
    """ 查看社团信息 """
    COLLECTION_CHARACTER = 0
    """ 收藏角色 """
    UN_COLLECTION_CHARACTER = 0
    """ 取消收藏 """
    COLLECTION_SYSTEM = 0
    """ 启用收藏模式 """
    UN_COLLECTION_SYSTEM = 0
    """ 关闭收藏模式 """
    SAVE = 0
    """ 读写存档 """
    DEBUG_ON = 0
    """ 开启debug模式 """
    DEBUG_OFF = 0
    """ 关闭debug模式 """


class Debug:
    """debug指令id"""

    """
    =========================
    系统类指令
    =========================
    """
    ADD_MONEY = 0
    """ 增加金钱 """
    ADD_TARGET_FAVORABILITY = 0
    """ 增加交互对象好感 """
    FORCE_HAND_IN_HAND = 0
    """ 强制牵手 """
    FORCE_MAKE_LOVE = 0
    """ 强制进入做爱状态 """
