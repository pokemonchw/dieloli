from types import FunctionType
from Script.Core import cache_control, game_type, get_text
from Script.Design import handle_debug, constant, character_handle
from Script.UI.Moudle import panel


cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """
_: FunctionType = get_text._
""" 翻译api """


@handle_debug.add_debug(
    constant.Debug.ADD_MONEY, constant.DebugInstructType.DEFAULT, _("增加金钱"), {}
)
def handle_add_money():
    """处理增加金钱指令"""
    now_panel = panel.AskNumberForOneMessage()
    now_panel.set(_("请输入想要增加的金钱数量"), 999999999)
    add_money = now_panel.draw()
    cache.character_data[0].money += add_money


@handle_debug.add_debug(
    constant.Debug.ADD_TARGET_FAVORABILITY, constant.DebugInstructType.DEFAULT, _("增加好感"), {constant.Premise.HAVE_TARGET}
)
def handle_add_favorability():
    """处理增加交互对象好感指令"""
    now_panel = panel.AskNumberForOneMessage()
    now_panel.set(_("请输入想要增加的好感数量"), 999999999)
    add_favorability = now_panel.draw()
    character_data: game_type.Character = cache.character_data[0]
    character_handle.add_favorability(0, character_data.target_character_id, add_favorability, None, cache.game_time)


@handle_debug.add_debug(
    constant.Debug.FORCE_HAND_IN_HAND, constant.DebugInstructType.DEFAULT, _("强制牵手"), {constant.Premise.HAVE_TARGET}
)
def handle_force_hand_in_hand():
    """处理强制牵手指令"""
    character_data: game_type.Character = cache.character_data[0]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    target_data.follow = 0
    character_data.pulling = target_data.cid


@handle_debug.add_debug(
    constant.Debug.FORCE_MAKE_LOVE, constant.DebugInstructType.DEFAULT, _("强制做爱"), {constant.Premise.HAVE_TARGET}
)
def handle_force_make_love():
    """处理强制做爱指令"""
    character_data: game_type.Character = cache.character_data[0]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    target_data.passive_sex = 1
