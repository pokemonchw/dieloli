from types import FunctionType
import datetime
from Script.Core import (
    cache_control, game_type, get_text,
    save_handle,
)
from Script.Design import (
    update, character, clothing,
    constant, handle_instruct,
    game_time, map_handle, character_handle,
    character_behavior, attr_calculation, weather,
)
from Script.Config import normal_config
from Script.UI.Model import draw


cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """
_: FunctionType = get_text._
""" 翻译api """
width: int = normal_config.config_normal.text_width
""" 屏幕宽度 """


@handle_instruct.add_instruct(constant.Instruct.REST, constant.InstructType.REST, _("休息"), {})
def handle_rest():
    """处理休息指令"""
    character.init_character_behavior_start_time(0, cache.game_time)
    character_data: game_type.Character = cache.character_data[0]
    character_data.behavior.duration = 10
    character_data.behavior.behavior_id = constant.Behavior.REST
    character_data.state = constant.CharacterStatus.STATUS_REST
    update.game_update_flow(10)


@handle_instruct.add_instruct(constant.Instruct.SIESTA, constant.InstructType.REST, _("午睡"),{constant.Premise.IN_SIESTA_TIME})
def handle_siesta():
    """ 处理午睡指令 """
    character.init_character_behavior_start_time(0, cache.game_time)
    character_data: game_type.Character = cache.character_data[0]
    character_data.behavior.duration = 30
    character_data.behavior.behavior_id = constant.Behavior.SIESTA
    character_data.behavior.behavior_id = constant.Behavior.SIESTA
    character_data.state = constant.CharacterStatus.STATUS_SIESTA
    update.game_update_flow(30)


@handle_instruct.add_instruct(
    constant.Instruct.SLEEP, constant.InstructType.REST, _("睡觉"), {constant.Premise.IN_DORMITORY}
)
def handle_sleep():
    """处理睡觉指令"""
    game_time.to_next_day()
    for character_id in cache.character_data:
        character_data = cache.character_data[character_id]
        # 若角色已经死亡则不做处理
        if character_data.state == constant.CharacterStatus.STATUS_DEAD:
            continue
        character_data.hit_point = character_data.hit_point_max
        character_data.mana_point = character_data.mana_point_max
        # 将角色送回宿舍
        character_position = cache.character_data[character_id].position
        character_dormitory = cache.character_data[character_id].dormitory
        character_dormitory = map_handle.get_map_system_path_for_str(character_dormitory)
        map_handle.character_move_scene(character_position, character_dormitory, character_id)
        # 增加角色身高
        _, growth_height = attr_calculation.predict_height(
            character_data.height.birth_height,
            character_data.height.expect_height,
            character_data.age,
            character_data.height.expect_age,
            character_data.sex
        )
        character_data.height.now_height += growth_height
        # 初始化角色状态
        character_data.state = constant.CharacterStatus.STATUS_ARDER
        if character_id:
            for status_id in character_data.status:
                character_data.status[status_id] = 0
        else:
            character_data.status[25] = 0
        # 初始化角色行为
        character_data.ai_target = 0
        character_data.behavior.behavior_id = constant.Behavior.SHARE_BLANKLY
        character_data.behavior.temporary_status = game_type.TemporaryStatus()
        character_data.behavior.start_time = cache.game_time
        character_data.behavior.move_src = []
        character_data.behavior.move_target = []
        character_data.behavior.course_id = 0
        character_data.behavior.duration = 0
        character_data.behavior.eat_food = None
        character_data.behavior.food_name = ""
        character_data.behavior.food_quality = 0
        last_hunger_time = character_data.last_hunger_time
        character_data.status.setdefault(27, 0)
        character_data.status.setdefault(28, 0)
        character_data.status[27] += 0.02 * (cache.game_time - last_hunger_time) / 60
        character_data.status[28] += 0.02 * (cache.game_time - last_hunger_time) / 60
        character_data.last_hunger_time = cache.game_time
        character_data.extreme_exhaustion_time = 0
        # 刷新角色体重和体脂率
        attr_calculation.handle_character_weight_and_body_fat(character_id)
    # 刷新食堂
    character_behavior.update_cafeteria()
    # 完成
    linefeed_draw = draw.NormalDraw()
    linefeed_draw.text = "\n"
    linefeed_draw.width = 1
    linefeed_draw.draw()
    now_draw = draw.LineFeedWaitDraw()
    now_draw.text = "萝莉祈祷中"
    now_draw.width = normal_config.config_normal.text_width
    now_draw.draw()
    weather.handle_weather()
    character_behavior.judge_character_dead(0)
    if not character_data.dead:
        save_handle.establish_save("auto")

