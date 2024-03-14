from types import FunctionType
from Script.Core import cache_control, game_type, get_text
from Script.Design import update, character, constant, handle_instruct, map_handle
from Script.Config import normal_config
from Script.UI.Moudle import draw


cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """
_: FunctionType = get_text._
""" 翻译api """
width: int = normal_config.config_normal.text_width
""" 屏幕宽度 """


@handle_instruct.add_instruct(
    constant.Instruct.TOUCH_HEAD,
    constant.InstructType.OBSCENITY,
    _("摸头"),
    {constant.Premise.HAVE_TARGET},
)
def handle_touch_head():
    """处理摸头指令"""
    character.init_character_behavior_start_time(0, cache.game_time)
    character_data = cache.character_data[0]
    character_data.behavior.duration = 2
    character_data.behavior.behavior_id = constant.Behavior.TOUCH_HEAD
    character_data.state = constant.CharacterStatus.STATUS_TOUCH_HEAD
    update.game_update_flow(2)


@handle_instruct.add_instruct(
    constant.Instruct.KISS,
    constant.InstructType.OBSCENITY,
    _("亲吻"),
    {constant.Premise.HAVE_TARGET},
)
def handle_kiss():
    """处理亲吻指令"""
    character.init_character_behavior_start_time(0, cache.game_time)
    character_data: game_type.Character = cache.character_data[0]
    character_data.behavior.duration = 2
    character_data.behavior.behavior_id = constant.Behavior.KISS
    character_data.state = constant.CharacterStatus.STATUS_KISS
    update.game_update_flow(2)


@handle_instruct.add_instruct(
    constant.Instruct.STROKE,
    constant.InstructType.OBSCENITY,
    _("抚摸"),
    {constant.Premise.HAVE_TARGET},
)
def handle_stroke():
    """处理抚摸指令"""
    character.init_character_behavior_start_time(0, cache.game_time)
    character_data: game_type.Character = cache.character_data[0]
    character_data.behavior.duration = 10
    character_data.behavior.behavior_id = constant.Behavior.STROKE
    character_data.state = constant.CharacterStatus.STATUS_STROKE
    update.game_update_flow(10)


@handle_instruct.add_instruct(
    constant.Instruct.TOUCH_CHEST,
    constant.InstructType.OBSCENITY,
    _("摸胸"),
    {constant.Premise.HAVE_TARGET},
)
def handle_touch_chest():
    """处理摸胸指令"""
    character.init_character_behavior_start_time(0, cache.game_time)
    character_data: game_type.Character = cache.character_data[0]
    character_data.behavior.duration = 10
    character_data.behavior.behavior_id = constant.Behavior.TOUCH_CHEST
    character_data.state = constant.CharacterStatus.STATUS_TOUCH_CHEST
    update.game_update_flow(10)


@handle_instruct.add_instruct(
    constant.Instruct.TARGET_UNDRESS,
    constant.InstructType.OBSCENITY,
    _("脱下对方衣服"),
    {constant.Premise.HAVE_TARGET},
)
def handle_target_undress():
    """处理脱下对方衣服指令"""
    character.init_character_behavior_start_time(0, cache.game_time)
    character_data: game_type.Character = cache.character_data[0]
    character_data.behavior.duration = 2
    character_data.behavior.behavior_id = constant.Behavior.TARGET_UNDRESS
    character_data.state = constant.CharacterStatus.STATUS_TARGET_UNDRESS
    update.game_update_flow(2)


@handle_instruct.add_instruct(
    constant.Instruct.STALKER,
    constant.InstructType.OBSCENITY,
    _("尾行"),
    {
        constant.Premise.HAVE_TARGET,
        constant.Premise.TARGET_IS_MOVING
    },
)
def handle_stalker():
    """ 处理尾行指令 """
    character.init_character_behavior_start_time(0, cache.game_time)
    character_data: game_type.Character = cache.character_data[0]
    accumulate_time = 0
    target_character_id = character_data.target_character_id
    while 1:
        target_data = cache.character_data[target_character_id]
        if target_data.state != constant.CharacterStatus.STATUS_MOVE:
            break
        if accumulate_time >= 30:
            now_draw = draw.LineFeedWaitDraw()
            now_draw.text = _("已尾行了半小时!")
            now_draw.width = normal_config.config_normal.text_width
            now_draw.draw()
        move_start_time = target_data.behavior.start_time
        move_end_time = target_data.behavior.duration + move_start_time
        need_time = move_end_time - cache.game_time
        if str(target_data.behavior.move_target) != str(character_data.position):
            character_data.behavior.duration = need_time
            character_data.behavior.behavior_id = constant.Behavior.MOVE
            character_data.behavior.move_src = character_data.position
            character_data.behavior.move_target = target_data.behavior.move_target
            character_data.state = constant.CharacterStatus.STATUS_MOVE
        update.game_update_flow(need_time+1)
        accumulate_time += need_time
    character_data.target_character_id = target_character_id
