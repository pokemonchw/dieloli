class StateMachine:
    """状态机id"""

    """
    =========================
    寻路类状态机
    =========================
    """
    MOVE_TO_CLASS = "move_to_class"
    """ 移动到所属教室 """
    MOVE_TO_OFFICEROOM = "move_to_officeroom"
    """ 移动至所属办公室 """
    MOVE_TO_NEAREST_CAFETERIA = "move_to_nearest_cafeteria"
    """移动至最近的取餐区"""
    MOVE_TO_NEAREST_RESTAURANT = "move_to_nearest_restaurant"
    """ 移动至最近的就餐区 """
    MOVE_TO_NEAREST_TOILET = "move_to_nearest_toilet"
    """ 移动至最近的洗手间 """
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
    MOVE_TO_DISLIKE_TARGET_SCENE = "move_to_dislike_target_scene"
    """ 移动至随机某个自己讨厌的人所在场景 """
    MOVE_TO_NOT_HAS_DISLIKE_TARGET_SCENE = "move_to_not_has_dislike_target_scene"
    """ 移动至没有自己讨厌的人的场景 """
    MOVE_TO_GROVE = "move_to_grove"
    """ 移动至小树林场景 """
    MOVE_TO_ITEM_SHOP = "move_to_item_shop"
    """ 移动至超市场景 """
    MOVE_TO_FOLLOW_TARGET_SCENE = "move_to_follow_target_scene"
    """ 移动至跟随对象所在场景 """
    MOVE_TO_LIBRARY = "move_to_library"
    """ 移动至图书馆 """
    MOVE_TO_SQUARE = "move_to_square"
    """ 移动至操场 """
    MOVE_TO_NO_MAN_SCENE = "move_to_no_man_scene"
    """ 移动至无人场景 """
    MOVE_TO_HAVE_CHARACTER_SCENE = "move_to_have_character_scene"
    """ 移动至有人的场景 """

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
    WEAR_CLEAN_COAT = "wear_clean_coat"
    """ 穿着干净的外套 """
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
    UNDRESS_COAT = "undress_coat"
    """ 脱掉外套 """
    TARGET_UNDRESS = "target_undress"
    """ 让交互对象脱掉衣服 """
    WEAR_CLOTHING = "wear_clothing"
    """ 穿上衣服 """
    UNDRESS = "undress"
    """ 脱掉衣服 """

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
    DANCE = "dance"
    """ 跳舞 """

    """
    =========================
    娱乐类状态机
    =========================
    """
    RUN = "run"
    """ 跑步 """


    """
    =========================
    互动类状态机
    =========================
    """
    CHAT_RAND_CHARACTER = "chat_rand_character"
    """ 和场景里随机对象闲聊 """
    CHAT_LIKE_CHARACTER = "chat_like_character"
    """ 和场景里自己喜欢的人聊天 """
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
    ABUSE_TO_DISLIKE_TARGET_IN_SCENE = "abuse_to_dislike_target_in_scene"
    """ 辱骂场景中自己讨厌的人 """
    ABUSE_NAKED_TARGET_IN_SCENE = "abuse_naked_target_in_scene"
    """ 辱骂场景中一丝不挂的人 """
    GENERAL_SPEECH = "general_speech"
    """ 发表演讲 """
    JOIN_CLUB_ACTIVITY = "join_club_activity"
    """ 参加社团活动 """

    """
    =========================
    科学类状态机
    =========================
    """
    SEE_STAR = "see_star"
    """ 看星星 """

    """
    =========================
    回复类状态机
    =========================
    """
    SLEEP = "sleep"
    """ 睡觉 """
    REST = "rest"
    """ 休息一会儿 """
    SIESTA = "siesta"
    """ 午睡 """

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
    MISSIONARY_POSITION = "missionary_position"
    """ 和交互对象正常体位做爱 """
