class TimeSlice:
    """ 时间段分类 """

    TIME_IN_CLASS = 0
    """ 上课中 """
    TIME_CLASS_INTERVAL = 1
    """ 两节课之间 """
    TIME_EARLY_READING = 2
    """ 早读 """
    TIME_RECESS = 3
    """ 大课间 """
    TIME_NOON_BREAK = 4
    """ 午休 """
    TIME_DINNER = 5
    """ 晚饭 """
    TIME_SLEEP = 6
    """ 晚休 """
    TIME_BREAKFAST = 7
    """ 早餐 """
    TIME_PLAYER = 8
    """ 娱乐 """


class CharacterStatus:
    """ 角色状态id """

    STATUS_ARDER = 0
    """ 休闲状态 """
    STATUS_MOVE = 1
    """ 移动状态 """
    STATUS_REST = 2
    """ 休息状态 """
    STATUS_ATTEND_CLASS = 3
    """ 上课状态 """
    STATUS_EAT = 4
    """ 进食状态 """


class Behavior:
    """ 行为id """

    SHARE_BLANKLY = 0
    """ 发呆 """
    MOVE = 1
    """ 移动 """
    REST = 2
    """ 休息 """
    ATTEND_CLASS = 3
    """ 上课 """
    EAT = 4
    """ 进食 """


class Panel:
    """ 面板id """

    TITLE = 0
    """ 标题面板 """
    CREATOR_CHARACTER = 1
    """ 创建角色面板 """
    GET_UP = 2
    """ 起床面板 """
    IN_SCENE = 3
    """ 场景互动面板 """
    SEE_MAP = 4
    """ 查看地图面板 """
    FOOD_SHOP = 5
    """ 食物商店面板 """


class Premise:
    """ 前提id """

    IN_CAFETERIA = 0
    """ 处于取餐区 """
    IN_RESTAURANT = 1
    """ 处于就餐区 """
    IN_BREAKFAST_TIME = 2
    """ 处于早餐时间段 """
    IN_LUNCH_TIME = 2
    """ 处于午餐时间段 """
    IN_DINNER_TIME = 3
    """ 处于晚餐时间段 """
    HUNGER = 4
    """ 处于饥饿状态 """
    HAVE_FOOD = 5
    """ 拥有食物 """
    NOT_HAVE_FOOD = 6
    """ 未拥有食物 """
    HAVE_TARGET = 7
    """ 拥有交互对象 """
    TARGET_NO_PLAYER = 8
    """ 交互对象不是玩家 """
    HAVE_DRAW_ITEM = 9
    """ 拥有绘画类道具 """
    HAVE_SHOOTING_ITEM = 10
    """ 拥有射击类道具 """
    HAVE_GUITAR = 11
    """ 拥有吉他 """
    HAVE_HARMONICA = 12
    """ 拥有口琴 """
    HAVE_BAM_BOO_FLUTE = 13
    """ 拥有竹笛 """
    HAVE_BASKETBALL = 14
    """ 拥有篮球 """
    HAVE_FOOTBALL = 15
    """ 拥有足球 """
    HAVE_TABLE_TENNIS = 16
    """ 拥有乒乓球 """
    IN_SWIMMING_POOL = 17
    """ 在游泳池中 """
    IN_CLASSROOM = 18
    """ 在教室中 """
    IS_STUDENT = 19
    """ 是学生 """
    IS_TEACHER = 20
    """ 是老师 """
    IN_SHOP = 21
    """ 在商店中 """
    IN_SLEEP_TIME = 22
    """ 处于睡觉时间 """
    IN_SIESTA_TIME = 23
    """ 处于午休时间 """
    TARGET_IS_FUTA_OR_WOMAN = 24
    """ 目标是扶她或女性 """
    TARGET_IS_FUTA_OR_MAN = 25
    """ 目标是扶她或男性 """
    IS_MAN = 26
    """ 角色是男性 """
    IS_WOMAN = 27
    """ 角色是女性 """
    TARGET_SAME_SEX = 28
    """ 目标与自身性别相同 """
    TARGET_AGE_SIMILAR = 29
    """ 目标与自身年龄相差不大 """
    TARGET_AVERAGE_HEIGHT_SIMILAR = 30
    """ 目标身高与平均身高相差不大 """
    TARGET_AVERAGE_HEIGHT_LOW = 31
    """ 目标身高低于平均身高 """
    TARGET_IS_PLAYER = 32
    """ 目标是玩家角色 """
    TARGET_AVERGAE_STATURE_SIMILAR = 33
    """ 目标体型与平均体型相差不大 """
    TARGET_NOT_PUT_ON_UNDERWEAR = 34
    """ 目标没穿上衣 """
    TARGET_NOT_PUT_ON_SKIRT = 35
    """ 目标没穿短裙 """
    IS_PLAYER = 36
    """ 是玩家角色 """
    NO_PLAYER = 37
    """ 不是玩家角色 """
    IN_PLAYER_SCENE = 38
    """ 与玩家处于相同场景 """
    LEAVE_PLAYER_SCENE = 39
    """ 离开玩家所在场景 """
    TARGET_IS_ADORE = 40
    """ 目标是爱慕对象 """
    TARGET_IS_ADMIRE = 41
    """ 目标是恋慕对象 """
    PLAYER_IS_ADORE = 42
    """ 玩家是爱慕对象 """
    EAT_SPRING_FOOD = 43
    """ 食用了春药品质的食物 """


class InstructType:
    """ 指令类型 """

    DIALOGUE = 0
    """ 对话 """
    ACTIVE = 1
    """ 主动 """
    PASSIVE = 2
    """ 被动 """
    PERFORM = 3
    """ 表演 """
    OBSCENITY = 4
    """ 猥亵 """
    PLAY = 5
    """ 娱乐 """
    BATTLE = 6
    """ 战斗 """
    STUDY = 7
    """ 学习 """
    REST = 8
    """ 休息 """


class Instruct:
    """ 指令id """

    REST = 0
    """ 休息 """
    BUY_FOOD = 1
    """ 购买食物 """
    EAT = 2
    """ 进食 """
    MOVE = 3
    """ 移动 """
    SEE_ATTR = 4
    """ 查看属性 """
    SEE_OWNER_ATTR = 5
    """ 查看自身属性 """
