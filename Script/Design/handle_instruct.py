from functools import wraps
from typing import Set
from types import FunctionType
from Script.Core import constant, cache_control, game_type, get_text
from Script.Design import update, character
from Script.UI.Panel import see_character_info_panel
from Script.Config import normal_config


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
    if instruct in cache.instruct_premise_data:
        cache.handle_instruct_data[instruct]()


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

        cache.handle_instruct_data[instruct_id] = return_wrapper
        cache.instruct_premise_data[instruct_id] = premise_set
        cache.instruct_type_data.setdefault(instruct_type, set())
        cache.instruct_type_data[instruct_type].add(instruct_id)
        cache.handle_instruct_name_data[instruct_id] = name
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
    constant.Instruct.SEE_ATTR, constant.InstructType.ACTIVE, _("查看属性"), {constant.Premise.HAVE_TARGET}
)
def handle_see_attr():
    """ 查看属性 """
    now_draw = see_character_info_panel.SeeCharacterInfoInScenePanel(
        cache.character_data[0].target_character_id, width
    )
    now_draw.draw()


@add_instruct(constant.Instruct.SEE_OWNER_ATTR, constant.InstructType.ACTIVE, _("查看自身属性"), {})
def handle_see_owner_attr():
    """ 查看自身属性 """
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
