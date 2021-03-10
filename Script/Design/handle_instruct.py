import random
import time
from functools import wraps
from typing import Set
from types import FunctionType
from Script.Core import constant, cache_control, game_type, get_text
from Script.Design import update, character, attr_calculation
from Script.UI.Panel import see_character_info_panel, see_save_info_panel
from Script.Config import normal_config
from Script.UI.Moudle import draw


cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """
_: FunctionType = get_text._
""" 翻译api """
width: int = normal_config.config_normal.text_width
""" 屏幕宽度 """


def handle_instruct(instruct: int):
    """
    处理执行指令
    Keyword arguments:
    instruct -- 指令id
    """
    if instruct in constant.instruct_premise_data:
        constant.handle_instruct_data[instruct]()


def add_instruct(instruct_id: int, instruct_type: int, name: str, premise_set: Set):
    """
    添加指令处理
    Keyword arguments:
    instruct_id -- 指令id
    instruct_type -- 指令类型
    name -- 指令绘制文本
    premise_set -- 指令所需前提集合
    """

    def decorator(func: FunctionType):
        @wraps(func)
        def return_wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        constant.handle_instruct_data[instruct_id] = return_wrapper
        constant.instruct_premise_data[instruct_id] = premise_set
        constant.instruct_type_data.setdefault(instruct_type, set())
        constant.instruct_type_data[instruct_type].add(instruct_id)
        constant.handle_instruct_name_data[instruct_id] = name
        return return_wrapper

    return decorator


@add_instruct(constant.Instruct.REST, constant.InstructType.REST, _("休息"), {})
def handle_rest():
    """ 处理休息指令 """
    character.init_character_behavior_start_time(0, cache.game_time)
    character_data = cache.character_data[0]
    character_data.behavior.duration = 10
    character_data.behavior.behavior_id = constant.Behavior.REST
    character_data.state = constant.CharacterStatus.STATUS_REST
    update.game_update_flow(10)


@add_instruct(
    constant.Instruct.BUY_FOOD, constant.InstructType.ACTIVE, _("购买食物"), {constant.Premise.IN_CAFETERIA}
)
def handle_buy_food():
    """ 处理购买食物指令 """
    cache.now_panel_id = constant.Panel.FOOD_SHOP


@add_instruct(constant.Instruct.EAT, constant.InstructType.ACTIVE, _("进食"), {constant.Premise.HAVE_FOOD})
def handle_eat():
    """ 处理进食指令 """
    cache.now_panel_id = constant.Panel.FOOD_BAG


@add_instruct(constant.Instruct.MOVE, constant.InstructType.ACTIVE, _("移动"), {})
def handle_move():
    """ 处理移动指令 """
    cache.now_panel_id = constant.Panel.SEE_MAP


@add_instruct(
    constant.Instruct.SEE_ATTR, constant.InstructType.SYSTEM, _("查看属性"), {constant.Premise.HAVE_TARGET}
)
def handle_see_attr():
    """ 查看属性 """
    see_character_info_panel.line_feed.draw()
    now_draw = see_character_info_panel.SeeCharacterInfoInScenePanel(
        cache.character_data[0].target_character_id, width
    )
    now_draw.draw()


@add_instruct(constant.Instruct.SEE_OWNER_ATTR, constant.InstructType.SYSTEM, _("查看自身属性"), {})
def handle_see_owner_attr():
    """ 查看自身属性 """
    see_character_info_panel.line_feed.draw()
    now_draw = see_character_info_panel.SeeCharacterInfoInScenePanel(0, width)
    now_draw.draw()


@add_instruct(
    constant.Instruct.CHAT, constant.InstructType.DIALOGUE, _("闲聊"), {constant.Premise.HAVE_TARGET}
)
def handle_chat():
    """ 处理闲聊指令 """
    character.init_character_behavior_start_time(0, cache.game_time)
    character_data = cache.character_data[0]
    character_data.behavior.duration = 10
    character_data.behavior.behavior_id = constant.Behavior.CHAT
    character_data.state = constant.CharacterStatus.STATUS_CHAT
    update.game_update_flow(10)


@add_instruct(
    constant.Instruct.BUY_ITEM, constant.InstructType.ACTIVE, _("购买道具"), {constant.Premise.IN_SHOP}
)
def handle_buy_item():
    """ 处理购买道具指令 """
    cache.now_panel_id = constant.Panel.ITEM_SHOP


@add_instruct(constant.Instruct.SINGING, constant.InstructType.PERFORM, _("唱歌"), {})
def handle_singing():
    """ 处理唱歌指令 """
    character.init_character_behavior_start_time(0, cache.game_time)
    character_data = cache.character_data[0]
    character_data.behavior.duration = 5
    character_data.behavior.behavior_id = constant.Behavior.SINGING
    character_data.state = constant.CharacterStatus.STATUS_SINGING
    update.game_update_flow(5)


@add_instruct(
    constant.Instruct.PLAY_PIANO,
    constant.InstructType.PERFORM,
    _("弹钢琴"),
    {constant.Premise.IN_MUSIC_CLASSROOM},
)
def handle_play_piano():
    """ 处理弹钢琴指令 """
    character.init_character_behavior_start_time(0, cache.game_time)
    character_data = cache.character_data[0]
    character_data.behavior.duration = 30
    character_data.behavior.behavior_id = constant.Behavior.PLAY_PIANO
    character_data.state = constant.CharacterStatus.STATUS_PLAY_PIANO
    update.game_update_flow(30)


@add_instruct(
    constant.Instruct.TOUCH_HEAD,
    constant.InstructType.OBSCENITY,
    _("摸头"),
    {constant.Premise.HAVE_TARGET},
)
def handle_touch_head():
    """ 处理摸头指令 """
    character.init_character_behavior_start_time(0, cache.game_time)
    character_data = cache.character_data[0]
    character_data.behavior.duration = 2
    character_data.behavior.behavior_id = constant.Behavior.TOUCH_HEAD
    character_data.state = constant.CharacterStatus.STATUS_TOUCH_HEAD
    update.game_update_flow(2)


@add_instruct(constant.Instruct.SAVE, constant.InstructType.SYSTEM, _("读写存档"), {})
def handle_save():
    """ 处理读写存档指令 """
    now_panel = see_save_info_panel.SeeSaveListPanel(width, 1)
    now_panel.draw()


@add_instruct(constant.Instruct.SLEEP, constant.InstructType.REST, _("睡觉"), {constant.Premise.IN_DORMITORY})
def handle_sleep():
    """ 处理睡觉指令 """
    character.init_character_behavior_start_time(0, cache.game_time)
    character_data: game_type.Character = cache.character_data[0]
    character_data.behavior.duration = 480
    character_data.behavior.behavior_id = constant.Behavior.SLEEP
    character_data.state = constant.CharacterStatus.STATUS_SLEEP
    cache.wframe_mouse.w_frame_skip_wait_mouse = 1
    update.game_update_flow(480)


@add_instruct(
    constant.Instruct.DRINK_SPRING, constant.InstructType.ACTIVE, _("喝泉水"), {constant.Premise.IN_FOUNTAIN}
)
def handle_drink_spring():
    """ 处理喝泉水指令 """
    value = random.randint(0, 100)
    now_draw = draw.WaitDraw()
    now_draw.width = width
    now_draw.text = "\n"
    character_data: game_type.Character = cache.character_data[0]
    if value <= 5 and not character_data.sex:
        now_draw.text += _("喝到了奇怪的泉水！身体变化了！！！")
        character_data.sex = 1
        character_data.height = attr_calculation.get_height(1, character_data.age)
        bmi = attr_calculation.get_bmi(character_data.weight_tem)
        character_data.weight = attr_calculation.get_weight(bmi, character_data.height.now_height)
        character_data.bodyfat = attr_calculation.get_body_fat(
            character_data.sex, character_data.bodyfat_tem
        )
        character_data.measurements = attr_calculation.get_measurements(
            character_data.sex,
            character_data.height.now_height,
            character_data.weight,
            character_data.bodyfat,
            character_data.bodyfat_tem,
        )
    else:
        now_draw.text += _("喝到了甜甜的泉水～")
        character_data.status[28] = 0
    now_draw.text += "\n"
    now_draw.draw()


@add_instruct(
    constant.Instruct.EMBRACE, constant.InstructType.ACTIVE, _("拥抱"), {constant.Premise.HAVE_TARGET}
)
def handle_embrace():
    """ 处理拥抱指令 """
    character.init_character_behavior_start_time(0, cache.game_time)
    character_data: game_type.Character = cache.character_data[0]
    character_data.behavior.duration = 3
    character_data.behavior.behavior_id = constant.Behavior.EMBRACE
    character_data.state = constant.CharacterStatus.STATUS_EMBRACE
    update.game_update_flow(3)


@add_instruct(
    constant.Instruct.KISS,
    constant.InstructType.OBSCENITY,
    _("亲吻"),
    {constant.Premise.HAVE_TARGET},
)
def handle_kiss():
    """ 处理亲吻指令 """
    character.init_character_behavior_start_time(0, cache.game_time)
    character_data: game_type.Character = cache.character_data[0]
    character_data.behavior.duration = 2
    character_data.behavior.behavior_id = constant.Behavior.KISS
    character_data.state = constant.CharacterStatus.STATUS_KISS
    update.game_update_flow(2)


@add_instruct(
    constant.Instruct.HAND_IN_HAND,
    constant.InstructType.ACTIVE,
    _("牵手"),
    {constant.Premise.HAVE_TARGET},
)
def handle_handle_in_handle():
    """ 处理牵手指令 """
    character.init_character_behavior_start_time(0, cache.game_time)
    character_data: game_type.Character = cache.character_data[0]
    character_data.behavior.duration = 10
    character_data.behavior.behavior_id = constant.Behavior.HAND_IN_HAND
    character_data.state = constant.CharacterStatus.STATUS_HAND_IN_HAND
    update.game_update_flow(10)


@add_instruct(
    constant.Instruct.STROKE,
    constant.InstructType.OBSCENITY,
    _("抚摸"),
    {constant.Premise.HAVE_TARGET},
)
def handle_stroke():
    """ 处理抚摸指令 """
    character.init_character_behavior_start_time(0, cache.game_time)
    character_data: game_type.Character = cache.character_data[0]
    character_data.behavior.duration = 10
    character_data.behavior.behavior_id = constant.Behavior.STROKE
    character_data.state = constant.CharacterStatus.STATUS_STROKE
    update.game_update_flow(10)


@add_instruct(
    constant.Instruct.TOUCH_CHEST,
    constant.InstructType.SEX,
    _("摸胸"),
    {constant.Premise.HAVE_TARGET},
)
def handle_touch_chest():
    """ 处理摸胸指令 """
    character.init_character_behavior_start_time(0, cache.game_time)
    character_data: game_type.Character = cache.character_data[0]
    character_data.behavior.duration = 10
    character_data.behavior.behavior_id = constant.Behavior.TOUCH_CHEST
    character_data.state = constant.CharacterStatus.STATUS_TOUCH_CHEST
    update.game_update_flow(10)


@add_instruct(
    constant.Instruct.COLLECTION_CHARACTER,
    constant.InstructType.SYSTEM,
    _("收藏角色"),
    {constant.Premise.TARGET_IS_NOT_COLLECTION, constant.Premise.TARGET_NO_PLAYER},
)
def handle_collection_character():
    """ 处理收藏角色指令 """
    character_data: game_type.Character = cache.character_data[0]
    target_character_id = character_data.target_character_id
    if target_character_id not in character_data.collection_character:
        character_data.collection_character.add(target_character_id)


@add_instruct(
    constant.Instruct.UN_COLLECTION_CHARACTER,
    constant.InstructType.SYSTEM,
    _("取消收藏"),
    {constant.Premise.TARGET_IS_COLLECTION, constant.Premise.TARGET_NO_PLAYER},
)
def handle_un_collection_character():
    """ 处理取消指令 """
    character_data: game_type.Character = cache.character_data[0]
    target_character_id = character_data.target_character_id
    if target_character_id in character_data.collection_character:
        character_data.collection_character.remove(target_character_id)


@add_instruct(
    constant.Instruct.COLLECTION_SYSTEM,
    constant.InstructType.SYSTEM,
    _("启用收藏模式"),
    {constant.Premise.UN_COLLECTION_SYSTEM},
)
def handle_collection_system():
    """ 处理启用收藏模式指令 """
    cache.is_collection = 1
    now_draw = draw.WaitDraw()
    now_draw.width = width
    now_draw.text = _("\n现在只会显示被收藏的角色的信息了！\n")
    now_draw.draw()


@add_instruct(
    constant.Instruct.UN_COLLECTION_SYSTEM,
    constant.InstructType.SYSTEM,
    _("关闭收藏模式"),
    {constant.Premise.IS_COLLECTION_SYSTEM},
)
def handle_un_collection_system():
    """ 处理关闭收藏模式指令 """
    cache.is_collection = 0
    now_draw = draw.WaitDraw()
    now_draw.width = width
    now_draw.text = _("\n现在会显示所有角色的信息了！\n")
    now_draw.draw()
