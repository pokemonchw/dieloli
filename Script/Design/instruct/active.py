import random
import datetime
from types import FunctionType
from Script.Core import cache_control, game_type, get_text
from Script.Design import (
    update, character, attr_calculation, constant, handle_instruct, character_move,
    map_handle, handle_achieve
)
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
    constant.Instruct.MOVE_TO_CAFETERIA,
    constant.InstructType.ACTIVE,
    _("去买吃的"),
    {
        constant.Premise.HUNGER,
        constant.Premise.NO_IN_CAFETERIA,
        constant.Premise.NOT_HAVE_FOOD,
    }
)
def handle_move_to_cafeteria():
    """处理去买吃的指令"""
    character_data: game_type.Character = cache.character_data[0]
    cafeteria_list = constant.place_data["Cafeteria"]
    now_position_str = map_handle.get_map_system_path_str_for_list(character_data.position)
    time_dict = {}
    for cafeteria in cafeteria_list:
        now_move_time = map_handle.scene_move_time[now_position_str][cafeteria]
        time_dict.setdefault(now_move_time, [])
        time_dict[now_move_time].append(cafeteria)
    min_time = min(time_dict.keys())
    to_cafeteria = map_handle.get_map_system_path_for_str(random.choice(time_dict[min_time]))
    character_move.own_charcter_move(map_handle.get_map_system_path_for_str(to_cafeteria))


@handle_instruct.add_instruct(
    constant.Instruct.ESCAPE_FROM_CROWD,
    constant.InstructType.ACTIVE,
    _("逃离人群"),
    {
        constant.Premise.HAS_NO_CHARACTER_SCENE,
        constant.Premise.SCENE_HAVE_OTHER_CHARACTER,
    }
)
def handle_escape_from_crowd():
    """处理逃离人群指令"""
    character_data: game_type.Character = cache.character_data[0]
    now_scene_str = map_handle.get_map_system_path_str_for_list(character_data.position)
    target_scene = cache.no_character_scene_set.pop()
    cache.no_character_scene_set.add(target_scene)
    character_move.own_charcter_move(map_handle.get_map_system_path_for_str(target_scene))


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
    now_draw = draw.WaitDraw()
    now_draw.width = width
    now_draw.text = "\n"
    character_data: game_type.Character = cache.character_data[0]
    value = random.randint(0, 100)
    if value <= 5 and not character_data.sex:
        now_draw.text += _("喝到了奇怪的泉水！身体变化了！！！")
        character_data.sex = 1
        character.init_character_end_age(0)
        character.init_character_height(0)
        character.init_character_weight_and_bodyfat(0)
        character.init_character_measurements(0)
        chest_tem = attr_calculation.get_rand_npc_chest_tem()
        character_data.chest_tem = chest_tem
        character_data.chest = attr_calculation.get_chest(chest_tem, character_data.birthday)
        cache_control.achieve.drowned_girl = True
    else:
        now_draw.text += _("喝到了甜甜的泉水～")
        character_data.status[28] = 0
    now_draw.text += "\n"
    now_draw.draw_event = True
    now_draw.draw()
    handle_achieve.check_all_achieve()


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
    if character_data.pulling != -1:
        target_data: game_type.Character = cache.character_data[character_data.pulling]
        target_data.follow = -1
        character_data.pulling = -1
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
    cache_control.achieve.first_wear_clothes = True
    handle_achieve.check_all_achieve()


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


@handle_instruct.add_instruct(
    constant.Instruct.SEE_STAR,
    constant.InstructType.ACTIVE,
    _("看星星"),
    {
        constant.Premise.IN_SLEEP_TIME,
        constant.Premise.OUT_DOOR
    },
)
def handle_see_star():
    """ 处理看星星指令 """
    character.init_character_behavior_start_time(0, cache.game_time)
    character_data: game_type.Character = cache.character_data[0]
    character_data.behavior.duration = 10
    character_data.behavior.behavior_id = constant.Behavior.SEE_STAR
    character_data.state = constant.CharacterStatus.STATUS_SEE_STAR
    update.game_update_flow(10)


@handle_instruct.add_instruct(
    constant.Instruct.CLUB_ACTIVITY,
    constant.InstructType.ACTIVE,
    _("社团活动"),
    {
        constant.Premise.IS_CLUB_ACTIVITY_TIME,
        constant.Premise.IN_CLUB_ACTIVITY_SCENE
    },
)
def handle_club_activity():
    """ 处理参加社团活动指令 """
    character.init_character_behavior_start_time(0, cache.game_time)
    character_data: game_type.Character = cache.character_data[0]
    identity_data: game_type.ClubIdentity = character_data.identity_data[2]
    club_data: game_type.ClubData = cache.all_club_data[identity_data.club_uid]
    now_time = datetime.datetime.fromtimestamp(character_data.behavior.start_time)
    now_week = now_time.weekday()
    week_time_dict = club_data.activity_time_dict[now_week]
    now_hour = now_time.hour
    hour_time_dict = week_time_dict[now_hour]
    now_minute = now_time.minute
    minute_time_dict = hour_time_dict[now_minute]
    activity_id = list(minute_time_dict.keys())[0]
    activity_data: game_type.ClubActivityData = club_data.activity_list[activity_id]
    character_data.behavior.behavior_id = activity_data.description
    character_data.behavior.duration = hour_time_dict[now_minute][activity_id]
    character_data.state = activity_data.description
    update.game_update_flow(character_data.behavior.duration+1)


@handle_instruct.add_instruct(
    constant.Instruct.SUICIDE,
    constant.InstructType.ACTIVE,
    _("信仰之跃"),
    {
        constant.Premise.IN_ROOFTOP_SCENE
    },
)
def handle_suicide():
    """ 处理自杀指令 """
    character_data: game_type.Character = cache.character_data[0]
    character_data.cause_of_death = 4
    character_data.dead = 1
    character_data.state = constant.CharacterStatus.STATUS_DEAD
    character_data.behavior.behavior_id = constant.Behavior.DEAD
    update.game_update_flow(1)
