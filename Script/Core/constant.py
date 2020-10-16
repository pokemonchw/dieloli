class FilePath:
    """  数据文件 """

    MESSAGE_PATH = "MessageList"
    """ 游戏系统消息文件路径 """
    CMD_PATH = "CmdText"
    """ 游戏命令菜单数据文件路径 """
    MENU_PATH = "MenuText"
    """ 游戏窗体按钮文本路径 """
    ROLE_PATH = "RoleAttributes"
    """ 角色属性描述文本路径(半废弃,待修正) """
    STAGE_WORD_PATH = "StageWord"
    """ 演出用语配置路径 """
    ERROR_PATH = "ErrorText"
    """ 错误信息文本路径 """
    ATTR_TEMPLATE_PATH = "AttrTemplate"
    """ 属性生成模板配置路径 """
    SYSTEM_TEXT_PATH = "SystemText"
    """ 系统信息文本路径 """
    NAME_LIST_PATH = "NameIndex"
    """ 名字权重数据路径 """
    FAMILY_NAME_LIST_PATH = "FamilyIndex"
    """ 姓氏权重数据路径 """
    FONT_CONFIG_PATH = "FontConfig"
    """ 富文本样式配置路径 """
    BAR_CONFIG_PATH = "BarConfig"
    """ 比例条数据配置路径 """
    PHASE_COURSE_PATH = "PhaseCourse"
    """ 各年级各科目权重数据配置路径 """
    COURSE_PATH = "Course"
    """ 各科目学习获得相关技能经验配置路径 """
    COURSE_SESSION_PATH = "CourseSession"
    """ 各学校上课时间配置路径 """
    KNOWLEDGE_PATH = "Knowledge"
    """ 通用技能相关数据配置路径 """
    LANGUAGE_SKILLS_PATH = "LanguageSkills"
    """ 语言技能相关配置路径 """
    EQUIPMENT_PATH = "Equipment"
    """ 服装类型数据配置路径 """
    STATURE_DESCRIPTION_PATH = "StatureDescription"
    """ 身材描述文本配置路径 """
    CHARACTER_STATE_PATH = "CharacterState"
    """ 角色状态数据结构配置路径 """
    WEAR_ITEM_PATH = "WearItem"
    """ 可穿戴道具数据配置路径 """
    NATURE_PATH = "Nature"
    """ 人格维度相关数据配置路径 """
    INSTRUCT_PATH = "Instruct"
    """ 指令显示需求数据配置路径 """
    FOOD_PATH = "Food"
    """ 食材配置路径 """
    RECIPES_PATH = "Recipes"
    """ 菜谱配置路径 """


class WindowMenu:
    """ 窗体菜单栏文本id """

    MENU_QUIT = "2"
    """ 退出 """
    MENU_SETTING = "3"
    """ 设置 """
    MENU_ABBOUT = "4"
    """ 关于 """
    MENU_FILE = "5"
    """ 文件 """
    MENU_OTHER = "6"
    """ 其他 """


class CmdMenu:
    """ 命令菜单相关 """

    LOGO_MENU = "logoMenu"
    """ 游戏主界面菜单 """
    CURRENCY_MENU = "currencyMenu"
    """ 通用菜单(半废弃,待修正) """
    SEX_MENU = "sexMenu"
    """ 性别选择菜单 """
    INPUT_NICK_NAME = "inputNickName"
    """ 昵称输入面板菜单 """
    INPUT_SELF_NEME = "inputSelfName"
    """ 自称输入面板菜单 """
    DETAILED_SETTING1 = "detailedSetting1"
    """ 创建角色时详细设置菜单1 """
    DETAILED_SETTING3 = "detailedSetting3"
    """ 创建角色时详细设置菜单3 """
    DETAILED_SETTING8 = "detailedSetting8"
    """ 创建角色时详细设置菜单8 """
    ACKNOWLEDGEMENT_ATTRIBUTE = "acknowledgmentAttribute"
    """ 创建角色时确认角色属性菜单 """
    MAIN_MENU = "mainMenu"
    """ 进入游戏后游戏主界面菜单 """
    SYSTEM_MENU = "systemMenu"
    """ 进入游戏后游戏主界面设置菜单 """
    SEE_ATTR_PANEL_HANDLE = "seeAttrPanelHandle"
    """ 查看角色面板控制菜单 """
    CHANGE_SAVE_PAGE = "changeSavePage"
    """ 切换存档页控制菜单 """
    SEE_ATTR_ON_EVERY_TIME = "seeAttrOnEveryTime"
    """ 查看角色属性的通用菜单 """
    SEE_CHARACTER_LIST = "seeCharacterList"
    """ 查看角色列表的控制菜单 """
    IN_SCENE_LIST1 = "inSceneList1"
    """ 场景中的控制菜单 """
    SEE_MAP = "seeMap"
    """ 查看地图面板的控制菜单 """
    GAME_HELP = "gameHelp"
    """ 游戏帮助面板的控制菜单 """
    SEE_CHARACTER_WEAR_CHOTHES = "seeCharacterWearClothes"
    """ 服装穿戴面板的控制菜单 """
    CHANGE_SCENE_CHARACTER_LIST = "changeSceneCharacterList"
    """ 切换场景中的角色列表页的控制菜单 """
    SEE_CHARACYTER_CLOTHES = "seeCharacterClothes"
    """ 查看服装列表的控制菜单 """
    ASK_SEE_CLOTHING_INFO_PANEL = "askSeeClothingInfoPanel"
    """ 查看或穿脱服装的请求菜单 """
    SEE_CLOTHING_INFO_ASK_PANEL = "seeClothingInfoAskPanel"
    """ 查看服装信息面板控制菜单 """
    SEE_KNOWLEDGE_ASK_PANEL = "seeKnowledgeAskPanel"
    """ 查看技能面板控制菜单 """
    ENTER_CHARACTER_NATURE = "enterCharacterNature"
    """ 确认属性面板控制菜单 """
    INSTRUCT_HEAD_PANEL = "instructHeadPanel"
    """ 操作指令过滤面板菜单 """
    INSTRUCT_PANEL = "instructPanel"
    """ 操作指令菜单 """
    BUY_FOOD_HEAD_PANEL = "buyFoodHeadPanel"
    """ 购买食物商店顶部面板 """
    BUY_FOOD_NOW_PANEL = "buyFoodNowPanel"
    """ 购买食物确认面板 """
    EAT_FOOD_NOW_PANEL = "eatFoodNowPanel"


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
    SEE_CHARACTER_INFO = 2
    """ 查看角色属性面板 """


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
