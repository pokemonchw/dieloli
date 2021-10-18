from typing import Dict, List, Set
from types import FunctionType


class CharacterStatus:
    """角色状态id"""

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
    STATUS_CHAT = 5
    """ 闲聊状态 """
    STATUS_PLAY_PIANO = 6
    """ 弹钢琴 """
    STATUS_SINGING = 7
    """ 唱歌 """
    STATUS_TOUCH_HEAD = 8
    """ 摸头 """
    STATUS_SLEEP = 9
    """ 睡觉 """
    STATUS_EMBRACE = 10
    """ 拥抱 """
    STATUS_KISS = 11
    """ 亲吻 """
    STATUS_HAND_IN_HAND = 12
    """ 牵手 """
    STATUS_DEAD = 13
    """ 死亡 """
    STATUS_STROKE = 14
    """ 抚摸 """
    STATUS_TOUCH_CHEST = 15
    """ 摸胸 """
    STATUS_TEACHING = 16
    """ 教学 """
    STATUS_PLAY_GUITAR = 17
    """ 弹吉他 """
    STATUS_SELF_STUDY = 18
    """ 自习 """
    STATUS_MASTURBATION = 19
    """ 手淫 """


class Behavior:
    """行为id"""

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
    CHAT = 5
    """ 闲聊 """
    PLAY_PIANO = 6
    """ 弹钢琴 """
    SINGING = 7
    """ 唱歌 """
    TOUCH_HEAD = 8
    """ 摸头 """
    SLEEP = 9
    """ 睡觉 """
    EMBRACE = 10
    """ 拥抱 """
    KISS = 11
    """ 亲吻 """
    HAND_IN_HAND = 12
    """ 牵手 """
    DEAD = 13
    """ 死亡 """
    STROKE = 14
    """ 抚摸 """
    TOUCH_CHEST = 15
    """ 摸胸 """
    TEACHING = 16
    """ 教学 """
    PLAY_GUITAR = 17
    """ 弹吉他 """
    SELF_STUDY = 18
    """ 自习 """
    MASTURBATION = 19
    """ 手淫 """


class StateMachine:
    """状态机id"""

    """
    =========================
    寻路类状态机
    =========================
    """
    MOVE_TO_CLASS = "move_to_class"
    """ 移动到所属教室 """
    MOVE_TO_NEAREST_CAFETERIA = "move_tp_nearest_cafeteria"
    """移动至最近的取餐区"""
    MOVE_TO_NEAREST_RESTAURANT = "move_to_nearest_restaurant"
    """ 移动至最近的就餐区 """
    MOVE_TO_MUSIC_ROOM = "move_to_music_room"
    """ 移动至音乐活动室 """
    MOVE_TO_DORMITORY = "move_to_dormitory"
    """ 移动至所属宿舍 """
    MOVE_TO_RAND_SCENE = "move_to_rand_scene"
    """ 移动至随机场景 """
    MOVE_TO_LIKE_TARGET_SCENE = "move_to_like_target_scene"
    """ 移动至随机某个自己喜欢的人所在场景 """
    MOVE_TO_NO_FIRST_KISS_LIKE_TARGET_SCENE = "move_to_no_first_kiss_like_target_scene"
    """ 移动至喜欢的还是初吻的人所在的场景 """
    MOVE_TO_GROVE = "move_to_grove"
    """ 移动至小树林场景 """
    MOVE_TO_ITEM_SHOP = "move_to_item_shop"
    """ 移动至超市场景 """
    MOVE_TO_FOLLOW_TARGET_SCENE = "move_to_follow_target_scene"
    """ 移动至跟随对象所在场景 """

    """
    =========================
    购物类状态机
    =========================
    """
    BUY_RAND_FOOD_AT_CAFETERIA = "buy_rand_food_at_cafeteria"
    """ 在取餐区购买随机食物 """
    BUY_RAND_DRINKS_AT_CAFETERIA = "buy_rand_drinks_at_cafeteria"
    """ 在取餐区购买随机饮料 """
    BUY_GUITAR = "buy_guitar"
    """ 购买吉他 """

    """
    =========================
    饮食类状态机
    =========================
    """
    EAT_BAG_RAND_FOOD = "eat_bag_rand_food"
    """ 食用背包内随机食物 """
    DRINK_RAND_DRINKS = "drink_rand_drinks"
    """ 饮用背包内随机饮料 """

    """
    =========================
    穿着类状态机
    =========================
    """
    WEAR_CLEAN_UNDERWEAR = "wear_clean_underwear"
    """ 穿干净的上衣 """
    WEAR_CLEAN_UNDERPANTS = "wear_clean_underpants"
    """ 穿干净的内裤 """
    WEAR_CLEAN_BRA = "wear_clean_bra"
    """ 穿干净的胸罩 """
    WEAR_CLEAN_PANTS = "wear_clean_pants"
    """ 穿干净的裤子 """
    WEAR_CLEAN_SKIRT = "wear_clean_skirt"
    """ 穿干净的短裙 """
    WEAR_CLEAN_SHOES = "wear_clean_shoes"
    """ 穿干净的鞋子 """
    WEAR_CLEAN_SOCKS = "wear_clean_socks"
    """ 穿干净的袜子 """
    UNDRESS_UNDERWEAR = "undress_underwear"
    """ 脱掉上衣 """
    UNDRESS_UNDERPANTS = "undress_underpants"
    """ 脱掉内裤 """
    UNDRESS_BRA = "undress_bra"
    """ 脱掉胸罩 """
    UNDRESS_PANTS = "undress_pants"
    """ 脱掉裤子 """
    UNDRESS_SKIRT = "undress_skirt"
    """ 脱掉短裙 """
    UNDRESS_SHOES = "undress_shoes"
    """ 脱掉鞋子 """
    UNDRESS_SOCKS = "undress_socks"
    """ 脱掉袜子 """
    WEAR_CLEAN_COAT = "wear_clean_coat"
    """ 穿着干净的外套 """
    UNDRESS_COAT = "undress_coat"
    """ 脱掉外套 """

    """
    =========================
    学习类状态机
    =========================
    """
    ATTEND_CLASS = "attend_class"
    """ 在教室上课 """
    TEACH_A_LESSON = "teach_a_lesson"
    """ 在教室教课 """
    SELF_STUDY = "self_study"
    """ 自习 """
    PLAY_PIANO = "play_piano"
    """ 弹钢琴 """
    SINGING = "singing"
    """ 唱歌 """
    PLAY_GUITAR = "play_guitar"
    """ 弹吉他 """

    """
    =========================
    互动类状态机
    =========================
    """
    CHAT_RAND_CHARACTER = "chat_rand_character"
    """ 和场景里随机对象闲聊 """
    SING_RAND_CHARACTER = "sing_rand_character"
    """ 唱歌给场景里随机对象听 """
    PLAY_PIANO_RAND_CHARACTER = "play_piano_rand_character"
    """ 弹奏钢琴给场景里随机对象听 """
    TOUCH_HEAD_TO_BEYOND_FRIENDSHIP_TARGET_IN_SCENE = (
        "touch_head_to_beyond_friendship_target_in_scene"
    )
    """ 对场景中抱有超越友谊想法的随机对象摸头 """
    EMBRACE_TO_BEYOND_FRIENDSHIP_TARGET_IN_SCENE = "embrace_to_beyond_friendship_target_in_scene"
    """ 对场景中抱有超越友谊想法的随机对象拥抱 """
    HAND_IN_HAND_TO_LIKE_TARGET_IN_SCENE = "hand_in_hand_to_like_target_in_scene"
    """ 牵住场景中自己喜欢的随机对象的手 """

    """
    =========================
    回复类状态机
    =========================
    """
    SLEEP = "sleep"
    """ 睡觉 """
    REST = "rest"
    """ 休息一会儿 """

    """
    =========================
    性爱类状态机
    =========================
    """
    KISS_TO_LIKE_TARGET_IN_SCENE = "kiss_to_like_target_in_scene"
    """ 和场景中自己喜欢的随机对象接吻 """
    KISS_TO_NO_FIRST_KISS_TARGET_IN_SCENE = "kiss_to_no_first_kiss_target_in_scene"
    """ 和场景中自己喜欢的还是初吻的随机对象接吻 """
    MASTURBATION = "masturbation"
    """ 手淫 """


class Panel:
    """面板id"""

    TITLE = 0
    """ 标题面板 """
    CREATOR_CHARACTER = 1
    """ 创建角色面板 """
    IN_SCENE = 2
    """ 场景互动面板 """
    SEE_MAP = 3
    """ 查看地图面板 """
    FOOD_SHOP = 4
    """ 食物商店面板 """
    FOOD_BAG = 5
    """ 食物背包面板 """
    ITEM_SHOP = 6
    """ 道具商店面板 """
    VIEW_SCHOOL_TIMETABLE = 7
    """ 查看课程表 """
    VIEW_CHARACTER_STATUS_LIST = 8
    """ 查看角色状态监控面板 """
    CLOTHING_SHOP = 9
    """ 服装商店面板 """


class Premise:
    """前提id"""

    """
    =========================
    角色身份类前提
    =========================
    """
    TARGET_NO_PLAYER = "target_no_player"
    """ 交互对象不是玩家 """
    IS_STUDENT = "is_student"
    """ 是学生 """
    IS_TEACHER = "is_teacher"
    """ 是老师 """
    TARGET_IS_PLAYER = "target_is_player"
    """ 目标是玩家角色 """
    IS_PLAYER = "is_player"
    """ 是玩家角色 """
    NO_PLAYER = "no_player"
    """ 不是玩家角色 """
    TARGET_IS_STUDENT = "target_is_student"
    """ 交互对象是学生 """
    IS_PRIMARY_SCHOOL_STUDENTS = "is_primary_school_students"
    """ 角色是小学生 """
    IS_PLAYER_TARGET = "is_player_target"
    """ 是玩家的交互对象 """

    """
    =========================
    角色性别类前提
    =========================
    """
    TARGET_IS_FUTA_OR_WOMAN = "target_is_futa_or_woman"
    """ 目标是扶她或女性 """
    TARGET_IS_FUTA_OR_MAN = "target_is_futa_or_man"
    """ 目标是扶她或男性 """
    IS_MAN = "is_man"
    """ 角色是男性 """
    IS_WOMAN = "is_woman"
    """ 角色是女性 """
    TARGET_SAME_SEX = "target_same_sex"
    """ 目标与自身性别相同 """
    TARGET_IS_WOMAN = "target_is_woman"
    """ 交互对象是女性 """
    TARGET_IS_MAN = "target_is_man"
    """ 交互对象是男性 """
    IS_MAN_OR_WOMAN = "is_man_or_woman"
    """ 角色是男性或女性 """
    IS_NOT_ASEXUAL = "is_not_asexual"
    """ 角色不是无性 """
    IS_FUTA_OR_WOMAN = "is_futa_or_woman"
    """ 角色是扶她或女性 """

    """
    =========================
    角色性格类前提
    =========================
    """
    IS_HUMOR_MAN = "is_humor_man"
    """ 是一个幽默的人 """
    IS_LIVELY = "is_lively"
    """ 是一个活跃的人 """
    IS_INFERIORITY = "is_inferiority"
    """ 是一个自卑的人 """
    IS_AUTONOMY = "is_autonomy"
    """ 是一个自律的人 """
    IS_SOLITARY = "is_solitary"
    """ 是一个孤僻的人 """
    IS_INDULGE = "is_indulge"
    """ 是一个放纵的人 """
    TARGET_IS_SOLITARY = "target_is_solitary"
    """ 交互对象是一个孤僻的人 """
    IS_ENTHUSIASM = "is_enthusiasm"
    """ 是一个热情的人 """
    TARGET_IS_ASTUTE = "target_is_astute"
    """ 交互对象是一个机敏的人 """
    TARGET_IS_INFERIORITY = "target_is_inferiority"
    """ 交互对象是一个自卑的人 """
    TARGET_IS_ENTHUSIASM = "target_is_enthusiasm"
    """ 交互对象是一个热情的人 """
    TARGET_IS_SELF_CONFIDENCE = "target_is_self_confidence"
    """ 交互对象是一个自信的人 """
    IS_ASTUTE = "is_astute"
    """ 是一个机敏的人 """
    TARGET_IS_HEAVY_FEELING = "target_is_heavy_feeling"
    """ 交互对象是一个重情的人 """
    IS_HEAVY_FEELING = "is_heavy_feeling"
    """ 是一个重情的人 """
    TARGET_IS_APATHY = "target_is_apathy"
    """ 交互对象是一个冷漠的人 """
    IS_STARAIGHTFORWARD = "is_staraightforward"
    """ 是一个爽直的人 """

    """
    =========================
    场景状态类前提
    =========================
    """
    IN_CAFETERIA = "in_cafeteria"
    """ 处于取餐区 """
    IN_RESTAURANT = "in_restaurant"
    """ 处于就餐区 """
    IN_SWIMMING_POOL = "in_swimming_pool"
    """ 在游泳池中 """
    IN_CLASSROOM = "in_classroom"
    """ 在教室中 """
    IN_SHOP = "in_shop"
    """ 在商店中 """
    IN_PLAYER_SCENE = "in_player_scene"
    """ 与玩家处于相同场景 """
    SCENE_HAVE_OTHER_CHARACTER = "scene_have_other_character"
    """ 场景中有自己外的其他角色 """
    IN_DORMITORY = "in_dormitory"
    """ 在宿舍中 """
    IN_MUSIC_CLASSROOM = "in_music_classroom"
    """ 处于音乐活动室 """
    SCENE_NO_HAVE_OTHER_CHARACTER = "scene_no_have_other_character"
    """ 场景中没有有自己外的其他角色 """
    SCENE_CHARACTER_ONLY_PLAYER_AND_ONE = "scene_character_only_player_and_one"
    """ 场景中只有包括玩家在内的两个角色 """
    BEYOND_FRIENDSHIP_TARGET_IN_SCENE = "beyond_friendship_target_in_scene"
    """ 对场景中某个角色抱有超越友谊的想法 """
    IN_FOUNTAIN = "in_fountain"
    """ 在喷泉场景 """
    HAVE_OTHER_TARGET_IN_SCENE = "have_other_target_in_scene"
    """ 场景中有自己和交互对象以外的其他人 """
    NO_HAVE_OTHER_TARGET_IN_SCENE = "no_have_other_target_in_scene"
    """ 场景中没有自己和交互对象以外的其他人 """
    HAVE_LIKE_TARGET_IN_SCENE = "have_like_target_in_scene"
    """ 场景中有喜欢的人 """
    HAVE_NO_FIRST_KISS_LIKE_TARGET_IN_SCENE = "have_no_first_kiss_like_target_in_scene"
    """ 有自己喜欢的还是初吻的人在场景中 """
    NO_IN_CLASSROOM = "no_in_classroom"
    """ 不在教室中 """
    TEACHER_NO_IN_CLASSROOM = "target_no_in_classroom"
    """ 角色所属班级的老师不在教室中 """
    TEACHER_IN_CLASSROOM = "teacher_in_classroom"
    """ 角色所属班级的老师在教室中 """
    IS_BEYOND_FRIENDSHIP_TARGET_IN_SCENE = "is_beyond_friendship_target_in_scene"
    """ 场景中有角色对自己抱有超越友谊的想法 """
    HAVE_STUDENTS_IN_CLASSROOM = "have_students_in_classroom"
    """ 有所教班级的学生在教室中 """
    IN_ROOFTOP_SCENE = "in_rooftop_scene"
    """ 处于天台场景 """
    IN_GROVE = "in_grove"
    """ 处于小树林场景 """
    NO_IN_GROVE = "no_in_grove"
    """ 未处于小树林场景 """
    NAKED_CHARACTER_IN_SCENE = "naked_character_in_scene"
    """ 场景中有人一丝不挂 """
    IN_ITEM_SHOP = "in_item_shop"
    """ 不在超市中 """
    NO_IN_ITEM_SHOP = "no_in_item_shop"
    """ 在超市中 """
    IN_STUDENT_UNION_OFFICE = "in_student_union_office"
    """ 在学生会办公室中 """
    NO_IN_CAFETERIA = "no_in_cafeteria"
    """ 未处于取餐区 """
    NO_IN_RESTAURANT = "no_in_restaurant"
    """ 未处于就餐区 """

    """
    =========================
    时间状态类前提
    =========================
    """
    IN_BREAKFAST_TIME = "in_breakfast_time"
    """ 处于早餐时间段 """
    IN_LUNCH_TIME = "in_lunch_time"
    """ 处于午餐时间段 """
    IN_DINNER_TIME = "in_dinner_time"
    """ 处于晚餐时间段 """
    IN_SLEEP_TIME = "in_sleep_time"
    """ 处于睡觉时间 """
    IN_SIESTA_TIME = "in_siesta_time"
    """ 处于午休时间 """
    NO_IN_SLEEP_TIME = "no_in_sleep_time"
    """ 不处于睡觉时间 """
    ATTEND_CLASS_TODAY = "attend_class_today"
    """ 今日需要上课 """
    APPROACHING_CLASS_TIME = "approaching_class_time"
    """ 临近上课时间 """
    IN_CLASS_TIME = "in_class_time"
    """ 处于上课时间 """
    TONIGHT_IS_FULL_MOON = "tonight_is_full_moon"
    """ 今夜是满月 """

    """
    =========================
    角色技能类前提
    =========================
    """
    EXCELLED_AT_PLAY_MUSIC = "excelled_at_play_music"
    """ 擅长演奏 """
    EXCELLED_AT_SINGING = "excelled_at_singing"
    """ 擅长演唱 """
    NO_EXCELLED_AT_SINGING = "no_excelled_at_singing"
    """ 不擅长演唱 """
    NO_EXCELLED_AT_PLAY_MUSIC = "no_excelled_at_play_music"
    """ 不擅长演奏 """
    TARGET_UNARMED_COMBAT_IS_HIGHT = "target_unarmed_combat_is_hight"
    """ 交互对象徒手格斗技能比自己高 """
    GOOD_AT_ELOQUENCE = "good_at_eloquence"
    """ 角色擅长口才 """
    GOOD_AT_LITERATURE = "good_at_literature"
    """ 角色擅长文学 """
    GOOD_AT_WRITING = "good_at_writing"
    """ 角色擅长写作 """
    GOOD_AT_DRAW = "good_at_draw"
    """ 角色擅长绘画 """
    GOOD_AT_ART = "good_at_art"
    """ 角色擅长艺术 """
    TARGET_LITTLE_KNOWLEDGE_OF_RELIGION = "target_little_knowledge_of_religion"
    """ 交互对象对宗教一知半解 """
    TARGET_LITTLE_KNOWLEDGE_OF_FAITH = "target_little_knowledge_of_faith"
    """ 交互对象对信仰一知半解 """
    TARGET_LITTLE_KNOWLEDGE_OF_ASTRONOMY = "target_little_knowledge_of_astronomy"
    """ 交互对象对天文学一知半解 """
    TARGET_LITTLE_KNOWLEDGE_OF_ASTROLOGY = "target_little_knowledge_of_astrology"
    """ 交互对象对占星学一知半解 """
    NO_GOOD_AT_ELOQUENCE = "no_good_at_eloquence"
    """ 角色不擅长口才 """

    """
    =========================
    角色状态类前提
    =========================
    """
    HAVE_TARGET = "have_target"
    """ 拥有交互对象 """
    HUNGER = "hunger"
    """ 处于饥饿状态 """
    EAT_SPRING_FOOD = "eat_spring_food"
    """ 食用了春药品质的食物 """
    ARROGANT_HEIGHT = "arrogant_height"
    """ 傲慢情绪高涨 """
    HYPOSTHENIA = "hyposthenia"
    """ 体力不足 """
    PHYSICAL_STRENGHT = "physical_strenght"
    """ 体力充沛 """
    TARGET_DISGUST_IS_HIGHT = "target_disgust_is_hight"
    """ 交互对象反感情绪高涨 """
    TARGET_LUST_IS_HIGHT = "target_lust_is_hight"
    """ 交互对象色欲高涨 """
    TARGET_IS_LIVE = "target_is_live"
    """ 交互对象未死亡 """
    THIRSTY = "thirsty"
    """ 处于口渴状态 """
    LUST_IS_HIGHT = "lust_is_hight"
    """ 角色色欲高涨 """
    LUST_IS_LOW = "lust_is_low"
    """ 角色色欲低下 """
    NO_FOLLOW = "no_follow"
    """ 角色未处于跟随状态 """
    IS_LOSE_FIRST_KISS = "is_lose_first_kiss"
    """ 角色正在失去初吻 """
    TARGET_IS_LOSE_FIRST_KISS = "target_is_lose_first_kiss"
    """ 交互对象正在失去初吻 """

    """
    =========================
    角色道具类前提
    =========================
    """
    HAVE_FOOD = "have_food"
    """ 拥有食物 """
    NOT_HAVE_FOOD = "not_have_food"
    """ 未拥有食物 """
    HAVE_DRAW_ITEM = "have_draw_item"
    """ 拥有绘画类道具 """
    HAVE_SHOOTING_ITEM = "have_shooting_item"
    """ 拥有射击类道具 """
    HAVE_GUITAR = "have_guitar"
    """ 拥有吉他 """
    NO_HAVE_GUITAR = "no_have_guitar"
    """ 未拥有吉他 """
    HAVE_HARMONICA = "have_harmonica"
    """ 拥有口琴 """
    HAVE_BAM_BOO_FLUTE = "have_bam_boo_flute"
    """ 拥有竹笛 """
    HAVE_BASKETBALL = "have_basketball"
    """ 拥有篮球 """
    HAVE_FOOTBALL = "have_football"
    """ 拥有足球 """
    HAVE_TABLE_TENNIS = "have_table_tennis"
    """ 拥有乒乓球 """
    HAVE_UNDERWEAR = "have_underwear"
    """ 拥有上衣 """
    HAVE_UNDERPANTS = "have_underpants"
    """ 拥有内裤 """
    HAVE_BRA = "have_bra"
    """ 拥有胸罩 """
    HAVE_PANTS = "have_pants"
    """ 拥有裤子 """
    HAVE_SKIRT = "have_skirt"
    """ 拥有短裙 """
    HAVE_SHOES = "have_shoes"
    """ 拥有鞋子 """
    HAVE_SOCKS = "have_socks"
    """ 拥有袜子 """
    HAVE_DRINKS = "have_drinks"
    """ 背包中有饮料 """
    NO_HAVE_DRINKS = "no_have_drinks"
    """ 背包中没有饮料 """
    HAVE_COAT = "have_coat"
    """ 角色拥有外套 """

    """
    =========================
    角色身材类前提
    =========================
    """
    TARGET_AGE_SIMILAR = "target_age_similar"
    """ 目标与自身年龄相差不大 """
    TARGET_AVERAGE_HEIGHT_SIMILAR = "target_average_height_similar"
    """ 目标身高与平均身高相差不大 """
    TARGET_AVERAGE_HEIGHT_LOW = "target_average_height_low"
    """ 目标身高低于平均身高 """
    TARGET_AVERGAE_STATURE_SIMILAR = "target_average_stature_similar"
    """ 目标体型与平均体型相差不大 """
    CHEST_IS_NOT_CLIFF = "chest_is_not_cliff"
    """ 胸围不是绝壁 """
    TARGET_HEIGHT_LOW = "target_height_low"
    """ 交互对象身高低于自身身高 """
    TARGET_IS_HEIGHT = "target_is_height"
    """ 目标比自己高 """
    TARGET_CHEST_IS_CLIFF = "target_chest_is_cliff"
    """ 交互对象胸围是绝壁 """
    TARGET_AVERAGE_STATURE_HEIGHT = "target_average_stature_height"
    """ 目标体型比平均体型更胖 """
    TARGET_CHEST_IS_NOT_CLIFF = "target_chest_is_not_chiff"
    """ 交互对象胸围不是绝壁 """

    """
    =========================
    角色穿着类前提
    =========================
    """
    TARGET_NOT_PUT_ON_UNDERWEAR = "target_not_put_on_underwear"
    """ 目标没穿上衣 """
    TARGET_NOT_PUT_ON_SKIRT = "target_not_put_on_skirt"
    """ 目标没穿短裙 """
    NO_WEAR_UNDERWEAR = "no_wear_underwear"
    """ 没穿上衣 """
    NO_WEAR_UNDERPANTS = "no_wear_underpants"
    """ 没穿内裤 """
    NO_WEAR_BRA = "no_wear_bra"
    """ 没穿胸罩 """
    NO_WEAR_PANTS = "no_wear_pants"
    """ 没穿裤子 """
    NO_WEAR_SKIRT = "no_wear_skirt"
    """ 没穿短裙 """
    NO_WEAR_SHOES = "no_wear_shoes"
    """ 没穿鞋子 """
    NO_WEAR_SOCKS = "no_wear_socks"
    """ 没穿袜子 """
    TARGET_IS_NAKED = "target_is_naked"
    """ 交互对象一丝不挂 """
    IS_NAKED = "is_naked"
    """ 角色一丝不挂 """
    NO_WEAR_COAT = "no_wear_coat"
    """ 没穿外套 """
    WEAR_COAT = "wear_coat"
    """ 角色穿了外套 """
    WEAR_UNDERWEAR = "wear_underwear"
    """ 角色穿了上衣 """
    WEAR_UNDERPANTS = "wear_underpants"
    """ 角色穿了内裤 """
    WEAR_BRA = "wear_bra"
    """ 角色穿了胸罩 """
    WEAR_PANTS = "wear_pants"
    """ 角色穿了裤子 """
    WEAR_SKIRT = "wear_skirt"
    """ 角色穿了短裙 """
    WEAR_SHOES = "wear_shoes"
    """ 角色穿了鞋子 """
    WEAR_SOCKS = "wear_socks"
    """ 角色穿了袜子 """

    """
    =========================
    角色性经验类前提
    =========================
    """
    IS_TARGET_FIRST_KISS = "is_target_first_kiss"
    """ 是交互对象的初吻对象 """
    SEX_EXPERIENCE_IS_HIGHT = "sex_experience_is_hight"
    """ 性技熟练 """
    RICH_EXPERIENCE_IN_SEX = "rich_experience_in_sex"
    """ 角色性经验丰富 """
    TARGET_NO_EXPERIENCE_IN_SEX = "target_no_experience_in_sex"
    """ 交互对象没有性经验 """
    NO_RICH_EXPERIENCE_IN_SEX = "no_rich_experience_in_sex"
    """ 角色性经验不丰富 """
    TARGET_NO_FIRST_KISS = "target_no_first_kiss"
    """ 交互对象初吻还在 """
    NO_FIRST_KISS = "no_first_kiss"
    """ 初吻还在 """
    TARGET_HAVE_FIRST_KISS = "target_have_first_kiss"
    """ 交互对象初吻不在了 """
    HAVE_FIRST_KISS = "have_first_kiss"
    """ 初吻不在了 """
    TARGET_NO_FIRST_HAND_IN_HAND = "target_no_first_hand_in_hand"
    """ 交互对象没有牵过手 """
    NO_FIRST_HAND_IN_HAND = "no_first_hand_in_hand"
    """ 没有牵过手 """
    HAVE_LIKE_TARGET_NO_FIRST_KISS = "have_like_target_no_first_kiss"
    """ 有自己喜欢的人的初吻还在 """
    TARGET_CLITORIS_LEVEL_IS_HIGHT = "target_clitoris_level_is_hight"
    """ 交互对象阴蒂开发度高 """

    """
    =========================
    角色好感类前提
    =========================
    """
    TARGET_IS_ADORE = "target_is_adore"
    """ 目标是爱慕对象 """
    TARGET_IS_ADMIRE = "target_is_admire"
    """ 目标是恋慕对象 """
    PLAYER_IS_ADORE = "player_is_adore"
    """ 玩家是爱慕对象 """
    TARGET_IS_BEYOND_FRIENDSHIP = "target_is_beyond_friendship"
    """ 对目标抱有超越友谊的想法 """
    IS_BEYOND_FRIENDSHIP_TARGET = "is_beyond_friendship_target"
    """ 目标对自己抱有超越友谊的想法 """
    TARGET_ADORE = "target_adore"
    """ 被交互对象爱慕 """
    NO_BEYOND_FRIENDSHIP_TARGET = "no_beyond_friendship"
    """ 目标对自己没有有超越友谊的想法 """
    TARGET_ADMIRE = "target_admire"
    """ 被交互对象恋慕 """
    HAVE_LIKE_TARGET = "have_like_target"
    """ 有喜欢的人 """

    """
    =========================
    角色行为类前提
    =========================
    """
    LEAVE_PLAYER_SCENE = "leave_player_scene"
    """ 离开玩家所在场景 """
    TARGET_IS_SLEEP = "target_is_sleep"
    """ 交互对象正在睡觉 """
    TARGET_IS_SING = "target_is_sing"
    """ 交互对象正在唱歌 """

    """
    =========================
    辅助系统类前提
    =========================
    """
    HAVE_FOLLOW = "have_follow"
    """ 角色拥有跟随对象 """
    NO_IN_FOLLOW_TARGET_SCENE = "no_in_follow_target_scene"
    """ 跟随的对象不在当前场景 """
    TARGET_IS_FOLLOW_PLAYER = "target_is_follow_player"
    """ 交互对象正在跟随玩家 """
    TARGET_NOT_FOLLOW_PLAYER = "target_not_follow_player"
    """ 交互对象未跟随玩家 """
    IS_COLLECTION_SYSTEM = "is_collection_system"
    """ 玩家已启用收藏模式 """
    UN_COLLECTION_SYSTEM = "un_collection_system"
    """ 玩家未启用收藏模式 """
    TARGET_IS_COLLECTION = "target_is_collection"
    """ 交互对象已被玩家收藏 """
    TARGET_IS_NOT_COLLECTION = "target_is_not_collection"
    """ 交互对象未被玩家收藏 """


class BehaviorEffect:
    """行为结算效果函数"""

    ADD_SMALL_HIT_POINT = 0
    """ 增加少量体力 """
    ADD_SMALL_MANA_POINT = 1
    """ 增加少量气力 """
    ADD_INTERACTION_FAVORABILITY = 2
    """ 增加基础互动好感 """
    SUB_SMALL_HIT_POINT = 3
    """ 减少少量体力 """
    SUB_SMALL_MANA_POINT = 4
    """ 减少少量气力 """
    MOVE_TO_TARGET_SCENE = 5
    """ 移动至目标场景 """
    EAT_FOOD = 6
    """ 食用指定食物 """
    ADD_SOCIAL_FAVORABILITY = 7
    """ 增加社交关系好感 """
    ADD_INTIMACY_FAVORABILITY = 8
    """ 增加亲密行为好感(关系不足2则增加反感) """
    ADD_INTIMATE_FAVORABILITY = 9
    """ 增加私密行为好感(关系不足3则增加反感) """
    ADD_SMALL_SING_EXPERIENCE = 10
    """ 增加少量唱歌技能经验 """
    ADD_SMALL_ELOQUENCE_EXPERIENCE = 11
    """ 增加少量口才技能经验 """
    ADD_SMALL_PLAY_MUSIC_EXPERIENCE = 12
    """ 增加少量演奏技能经验 """
    ADD_SMALL_PERFORM_EXPERIENCE = 13
    """ 增加少量表演技能经验 """
    ADD_SMALL_CEREMONY_EXPERIENCE = 14
    """ 增加少量礼仪技能经验 """
    ADD_SMALL_SEX_EXPERIENCE = 15
    """ 增加少量性爱技能经验 """
    ADD_SMALL_MOUTH_SEX_EXPERIENCE = 16
    """ 增加少量嘴部性爱经验 """
    ADD_SMALL_MOUTH_HAPPY = 17
    """ 增加少量嘴部快感 """
    FIRST_KISS = 18
    """ 记录初吻 """
    FIRST_HAND_IN_HAND = 19
    """ 记录初次牵手 """
    ADD_MEDIUM_HIT_POINT = 20
    """ 增加中量体力 """
    ADD_MEDIUM_MANA_POINT = 21
    """ 增加中量气力 """
    TARGET_ADD_SMALL_CHEST_SEX_EXPERIENCE = 22
    """ 交互对象增加少量胸部性爱经验 """
    TARGET_ADD_SMALL_CHEST_HAPPY = 23
    """ 交互对象增加少量胸部快感 """
    TARGET_ADD_SMALL_CLITORIS_SEX_EXPERIENCE = 24
    """ 交互对象增加少量阴蒂性爱经验 """
    TARGET_ADD_SMALL_PENIS_SEX_EXPERIENCE = 25
    """ 交互对象增加少量阴茎性爱经验 """
    TARGET_ADD_SMALL_CLITORIS_HAPPY = 26
    """ 交互对象增加少量阴蒂快感 """
    TARGET_ADD_SMALL_PENIS_HAPPY = 27
    """ 交互对象增加少量阴茎经验 """
    ADD_SMALL_LUST = 28
    """ 自身增加少量色欲 """
    TARGET_ADD_SMALL_LUST = 29
    """ 交互对象增加少量色欲 """
    INTERRUPT_TARGET_ACTIVITY = 30
    """ 打断交互对象活动 """
    TARGET_ADD_SMALL_ELOQUENCE_EXPERIENCE = 31
    """ 交互对象增加少量口才技能经验 """
    TARGET_ADD_FAVORABILITY_FOR_ELOQUENCE = 32
    """ 按口才技能增加交互对象好感 """
    ADD_SMALL_ATTEND_CLASS_EXPERIENCE = 33
    """ 按学习课程增加少量对应技能经验 """
    ADD_STUDENTS_COURSE_EXPERIENCE_FOR_IN_CLASS_ROOM = 34
    """ 按课程增加教室内本班级学生的技能经验 """
    TARGET_ADD_FAVORABILITY_FOR_PERFORMANCE = 35
    """ 按表演技能增加交互对象好感 """
    TARGET_ADD_FAVORABILITY_FOR_SING = 36
    """ 按演唱技能增加交互对象好感 """
    TARGET_ADD_FAVORABILITY_FOR_PLAY_MUSIC = 37
    """ 按演奏技能增加交互对象好感 """
    TARGET_ADD_FAVORABILITY_FOR_TARGET_INTEREST = 38
    """ 按交互对象兴趣增加交互对象好感 """
    ADD_SMALL_CLITORIS_SEX_EXPERIENCE = 39
    """ 增加少量阴蒂性爱经验 """
    ADD_SMALL_PENIS_SEX_EXPERIENCE = 40
    """ 增加少量阴茎性爱经验 """
    ADD_SMALL_CLITORIS_HAPPY = 41
    """ 增加少量阴蒂快感 """
    ADD_SMALL_PENIS_HAPPY = 42
    """ 增加少量阴茎快感 """
    TARGET_FOLLOW_SELF = 43
    """ 交互对象跟随自己 """
    UNFOLLOW = 44
    """ 取消跟随状态 """


class InstructType:
    """指令类型"""

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
    SEX = 9
    """ 性爱 """
    SYSTEM = 10
    """ 系统 """


class Instruct:
    """指令id"""

    CHAT = 0
    """ 闲聊 """
    REST = 0
    """ 休息 """
    SLEEP = 0
    """ 睡觉 """
    SINGING = 0
    """ 唱歌 """
    PLAY_PIANO = 0
    """ 弹钢琴 """
    PLAY_GUITAR = 0
    """ 弹吉他 """
    TOUCH_HEAD = 0
    """ 摸头 """
    TOUCH_CHEST = 0
    """ 摸胸 """
    STROKE = 0
    """ 抚摸 """
    HAND_IN_HAND = 0
    """ 牵手 """
    LET_GO = 0
    """ 放手 """
    EMBRACE = 0
    """ 拥抱 """
    KISS = 0
    """ 亲吻 """
    MASTURBATION = 0
    """ 手淫 """
    EAT = 0
    """ 进食 """
    DRINK_SPRING = 0
    """ 喝泉水 """
    BUY_FOOD = 0
    """ 购买食物 """
    BUY_ITEM = 0
    """ 购买道具 """
    BUY_CLOTHING = 0
    """ 购买服装 """
    MOVE = 0
    """ 移动 """
    ATTEND_CLASS = 0
    """ 上课 """
    TEACH_A_LESSON = 0
    """ 教课 """
    SELF_STUDY = 0
    """ 自习 """
    VIEW_THE_SCHOOL_TIMETABLE = 0
    """ 查看课程表 """
    SEE_ATTR = 0
    """ 查看属性 """
    SEE_OWNER_ATTR = 0
    """ 查看自身属性 """
    VIEW_CHARACTER_STATUS_LIST = 0
    """ 查看角色状态列表 """
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


i = 0
for k in Instruct.__dict__:
    if isinstance(Instruct.__dict__[k], int):
        setattr(Instruct, k, i)
        i += 1


handle_premise_data: Dict[str, FunctionType] = {}
""" 前提处理数据 """
handle_instruct_data: Dict[int, FunctionType] = {}
""" 指令处理数据 """
handle_instruct_name_data: Dict[int, str] = {}
""" 指令对应文本 """
instruct_type_data: Dict[int, Set] = {}
""" 指令类型拥有的指令集合 """
instruct_premise_data: Dict[int, Set] = {}
""" 指令显示的所需前提集合 """
handle_state_machine_data: Dict[int, FunctionType] = {}
""" 角色状态机函数 """
family_region_list: Dict[int, str] = {}
""" 姓氏区间数据 """
boys_region_list: Dict[int, str] = {}
""" 男孩名字区间数据 """
girls_region_list: Dict[int, str] = {}
""" 女孩名字区间数据 """
family_region_int_list: List[int] = []
""" 姓氏权重区间数据 """
boys_region_int_list: List[int] = []
""" 男孩名字权重区间数据 """
girls_region_int_list: List[int] = []
""" 女孩名字权重区间数据 """
panel_data: Dict[int, FunctionType] = {}
"""
面板id对应的面板绘制函数集合
面板id:面板绘制函数对象
"""
place_data: Dict[str, List[str]] = {}
""" 按房间类型分类的场景列表 场景标签:场景路径列表 """
cmd_map: Dict[int, FunctionType] = {}
""" cmd存储 """
settle_behavior_effect_data: Dict[int, FunctionType] = {}
""" 角色行为结算处理器 处理器id:处理器 """
