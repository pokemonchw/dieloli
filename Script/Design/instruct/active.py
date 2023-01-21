import random
from types import FunctionType
from Script.Core import cache_control, game_type, get_text
from Script.Design import update, character, attr_calculation, constant, handle_instruct
from Script.Config import normal_config
from Script.UI.Moudle import draw


cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """
_: FunctionType = get_text._
""" 翻译api """
width: int = normal_config.config_normal.text_width
""" 屏幕宽度 """


@handle_instruct.add_instruct(
    constant.Instruct.EAT, constant.InstructType.ACTIVE, _("进食"), {constant.Premise.HAVE_FOOD}
)
def handle_eat():
    """处理进食指令"""
    cache.now_panel_id = constant.Panel.FOOD_BAG


@handle_instruct.add_instruct(constant.Instruct.MOVE, constant.InstructType.ACTIVE, _("移动"), {})
def handle_move():
    """处理移动指令"""
    cache.now_panel_id = constant.Panel.SEE_MAP


@handle_instruct.add_instruct(
    constant.Instruct.BUY_ITEM, constant.InstructType.ACTIVE, _("购买道具"), {constant.Premise.IN_SHOP}
)
def handle_buy_item():
    """处理购买道具指令"""
    cache.now_panel_id = constant.Panel.ITEM_SHOP


@handle_instruct.add_instruct(
    constant.Instruct.BUY_FOOD,
    constant.InstructType.ACTIVE,
    _("购买食物"),
    {constant.Premise.IN_CAFETERIA},
)
def handle_buy_food():
    """处理购买食物指令"""
    cache.now_panel_id = constant.Panel.FOOD_SHOP


@handle_instruct.add_instruct(
    constant.Instruct.BUY_CLOTHING,
    constant.InstructType.ACTIVE,
    _("购买服装"),
    {constant.Premise.IN_SHOP},
)
def handle_but_clothing():
    """处理购买服装指令"""
    cache.now_panel_id = constant.Panel.CLOTHING_SHOP


@handle_instruct.add_instruct(
    constant.Instruct.DRINK_SPRING,
    constant.InstructType.ACTIVE,
    _("喝泉水"),
    {constant.Premise.IN_FOUNTAIN},
)
def handle_drink_spring():
    """处理喝泉水指令"""
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
            character_data.bodyfat_tem,
        )
    else:
        now_draw.text += _("喝到了甜甜的泉水～")
        character_data.status[28] = 0
    now_draw.text += "\n"
    now_draw.draw()


@handle_instruct.add_instruct(
    constant.Instruct.EMBRACE, constant.InstructType.ACTIVE, _("拥抱"), {constant.Premise.HAVE_TARGET}
)
def handle_embrace():
    """处理拥抱指令"""
    character.init_character_behavior_start_time(0, cache.game_time)
    character_data: game_type.Character = cache.character_data[0]
    character_data.behavior.duration = 3
    character_data.behavior.behavior_id = constant.Behavior.EMBRACE
    character_data.state = constant.CharacterStatus.STATUS_EMBRACE
    update.game_update_flow(3)


@handle_instruct.add_instruct(
    constant.Instruct.HAND_IN_HAND,
    constant.InstructType.ACTIVE,
    _("牵手"),
    {constant.Premise.HAVE_TARGET, constant.Premise.TARGET_NOT_FOLLOW_PLAYER},
)
def handle_handle_in_handle():
    """处理牵手指令"""
    character.init_character_behavior_start_time(0, cache.game_time)
    character_data: game_type.Character = cache.character_data[0]
    character_data.behavior.duration = 10
    character_data.behavior.behavior_id = constant.Behavior.HAND_IN_HAND
    character_data.state = constant.CharacterStatus.STATUS_HAND_IN_HAND
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    target_data.follow = 0
    update.game_update_flow(10)


@handle_instruct.add_instruct(
    constant.Instruct.LET_GO,
    constant.InstructType.ACTIVE,
    _("放手"),
    {constant.Premise.TARGET_IS_FOLLOW_PLAYER},
)
def handle_let_go():
    """处理放手指令"""
    character_data: game_type.Character = cache.character_data[0]
    if character_data.target_character_id:
        target_data: game_type.Character = cache.character_data[character_data.target_character_id]
        target_data.follow = -1


@handle_instruct.add_instruct(
    constant.Instruct.WEAR,
    constant.InstructType.ACTIVE,
    _("穿衣"),
    {constant.Premise.IN_DORMITORY},
)
def handle_wear():
    """处理穿衣指令"""
    character.init_character_behavior_start_time(0, cache.game_time)
    character_data: game_type.Character = cache.character_data[0]
    character_data.behavior.duration = 2
    character_data.behavior.behavior_id = constant.Behavior.WEAR
    character_data.state = constant.CharacterStatus.STATUS_WEAR
    update.game_update_flow(2)


@handle_instruct.add_instruct(
    constant.Instruct.SELF_UNDRESS,
    constant.InstructType.ACTIVE,
    _("脱下自身衣物"),
    {constant.Premise.NOT_NAKED},
)
def handle_self_undress():
    """处理脱下自身衣物指令"""
    character.init_character_behavior_start_time(0, cache.game_time)
    character_data: game_type.Character = cache.character_data[0]
    character_data.behavior.duration = 2
    character_data.behavior.behavior_id = constant.Behavior.UNDRESS
    character_data.state = constant.CharacterStatus.STATUS_UNDRESS
    update.game_update_flow(2)
