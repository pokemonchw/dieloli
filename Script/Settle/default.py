import datetime
from types import FunctionType
from Script.Design import settle_behavior, character, character_handle, map_handle
from Script.Core import cache_control, constant, game_type, get_text
from Script.Config import game_config, normal_config
from Script.UI.Moudle import draw


_: FunctionType = get_text._
""" 翻译api """
window_width: int = normal_config.config_normal.text_width
""" 窗体宽度 """
cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_SMALL_HIT_POINT)
def handle_add_small_hit_point(
    character_id: int, add_time: int, change_data: game_type.CharacterStatusChange
):
    """
    增加少量体力
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    """
    character_data: game_type.Character = cache.character_data[character_id]
    add_hit_point = add_time
    character_data.hit_point += add_hit_point
    if character_data.hit_point > character_data.hit_point_max:
        add_hit_point -= character_data.hit_point - character_data.hit_point_max
        character_data.hit_point = character_data.hit_point_max
    change_data.hit_point += add_hit_point


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_SMALL_MANA_POINT)
def handle_add_small_mana_point(
    character_id: int, add_time: int, change_data: game_type.CharacterStatusChange
):
    """
    增加少量气力
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    """
    character_data: game_type.Character = cache.character_data[character_id]
    add_mana_point = add_time * 1.5
    character_data.mana_point += add_mana_point
    if character_data.mana_point > character_data.mana_point_max:
        add_mana_point -= character_data.mana_point - character_data.mana_point_max
        character_data.mana_point = character_data.mana_point_max
    change_data.mana_point += add_mana_point


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_INTERACTION_FAVORABILITY)
def handle_add_interaction_favoravility(
    character_id: int, add_time: int, change_data: game_type.CharacterStatusChange
):
    """
    增加基础互动好感
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id != character_id:
        target_data: game_type.Character = cache.character_data[character_data.target_character_id]
        change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
        target_change = change_data.target_change[target_data.cid]
        add_favorability = character.calculation_favorability(character_id, target_data.cid, add_time)
        character_handle.add_favorability(character_id, target_data.cid, add_favorability, target_change)


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.SUB_SMALL_HIT_POINT)
def handle_sub_small_hit_point(
    character_id: int, add_time: int, change_data: game_type.CharacterStatusChange
):
    """
    减少少量体力
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.hit_point -= add_time
    change_data.hit_point -= add_time


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.SUB_SMALL_MANA_POINT)
def handle_sub_small_mana_point(
    character_id: int, add_time: int, change_data: game_type.CharacterStatusChange
):
    """
    减少少量气力
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    """
    sub_mana = add_time * 1.5
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.mana_point >= sub_mana:
        character_data.mana_point -= sub_mana
        change_data.mana_point -= sub_mana
    else:
        change_data.mana_point -= character_data.mana_point
        sub_mana -= character_data.mana_point
        character_data.mana_point = 0
        character_data.hit_point -= sub_mana / 1.5
        change_data.hit_point -= sub_mana / 1.5


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.MOVE_TO_TARGET_SCENE)
def handle_move_to_target_scene(
    character_id: int, add_time: int, change_data: game_type.CharacterStatusChange
):
    """
    移动至目标场景
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if len(character_data.behavior.move_target):
        map_handle.character_move_scene(
            character_data.position, character_data.behavior.move_target, character_id
        )


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.EAT_FOOD)
def handle_eat_food(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange):
    """
    食用指定食物
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.behavior.eat_food != None:
        food: game_type.Food = character_data.behavior.eat_food
        eat_weight = 100
        if food.weight < eat_weight:
            eat_weight = food.weight
        for feel in food.feel:
            now_feel_value = food.feel[feel]
            now_feel_value = now_feel_value / food.weight
            now_feel_value *= eat_weight
            character_data.status.setdefault(feel, 0)
            change_data.status.setdefault(feel, 0)
            if feel in {27, 28}:
                character_data.status[feel] -= now_feel_value
                if character_data.status[feel] < 0:
                    character_data.status[feel] = 0
                change_data.status[feel] -= now_feel_value
            else:
                character_data.status[feel] += now_feel_value
                change_data.status[feel] += now_feel_value
        food.weight -= eat_weight
        food_name = ""
        if food.recipe == -1:
            food_config = game_config.config_food[food.id]
            food_name = food_config.name
        else:
            food_name = cache.recipe_data[food.recipe].name
        character_data.behavior.food_name = food_name
        character_data.behavior.food_quality = food.quality
        if food.weight <= 0:
            del character_data.food_bag[food.uid]


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_SOCIAL_FAVORABILITY)
def handle_add_social_favorability(
    character_id: int, add_time: int, change_data: game_type.CharacterStatusChange
):
    """
    增加社交关系好感
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id != character_id:
        target_data: game_type.Character = cache.character_data[character_data.target_character_id]
        if (
            character_id in target_data.social_contact_data
            and target_data.social_contact_data[character_id]
        ):
            change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
            target_change = change_data.target_change[target_data.cid]
            add_favorability = character.calculation_favorability(character_id, target_data.cid, add_time)
            add_favorability *= target_data.social_contact_data[character_id]
            if add_favorability:
                character_handle.add_favorability(
                    character_id, target_data.cid, add_favorability, target_change
                )


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_INTIMACY_FAVORABILITY)
def handle_add_intimacy_favorability(
    character_id: int, add_time: int, change_data: game_type.CharacterStatusChange
):
    """
    增加亲密行为好感
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id != character_id:
        target_data: game_type.Character = cache.character_data[character_data.target_character_id]
        add_favorability = character.calculation_favorability(
            character_id, character_data.target_character_id, add_time * 1.5
        )
        add_favorability_coefficient = add_favorability / (add_time * 1.5)
        social = target_data.social_contact_data[character_id]
        change_data.target_change.setdefault(character_data.target_character_id, game_type.TargetChange())
        target_change = change_data.target_change[target_data.cid]
        target_data: game_type.Character = cache.character_data[character_data.target_character_id]
        if (
            character_id in target_data.social_contact_data
            and target_data.social_contact_data[character_id] >= 2
        ):
            add_favorability += add_favorability_coefficient * social
            character_handle.add_favorability(
                character_id, target_data.cid, add_favorability, target_change
            )
        else:
            add_favorability -= add_favorability_coefficient * social
            cal_social = social
            if not cal_social:
                cal_social = 1
            add_disgust = (100 - add_favorability) / cal_social
            target_data.status.setdefault(12, 0)
            target_data.status[12] += add_disgust
            character_handle.add_favorability(
                character_id, target_data.cid, add_favorability, target_change
            )


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_INTIMATE_FAVORABILITY)
def handle_add_intimate_favorability(
    character_id: int, add_time: int, change_data: game_type.CharacterStatusChange
):
    """
    增加私密行为好感
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id != character_id:
        target_data: game_type.Character = cache.character_data[character_data.target_character_id]
        add_favorability = character.calculation_favorability(
            character_id, character_data.target_character_id, add_time * 2
        )
        add_favorability_coefficient = add_favorability / (add_time * 2)
        social = target_data.social_contact_data[character_id]
        change_data.target_change.setdefault(character_data.target_character_id, game_type.TargetChange())
        target_change = change_data.target_change[target_data.cid]
        target_data: game_type.Character = cache.character_data[character_data.target_character_id]
        if (
            character_id in target_data.social_contact_data
            and target_data.social_contact_data[character_id] >= 3
        ):
            add_favorability += add_favorability_coefficient * social
            character_handle.add_favorability(
                character_id, target_data.cid, add_favorability, target_change
            )
        else:
            add_favorability -= add_favorability_coefficient * social
            cal_social = social
            if not cal_social:
                cal_social = 1
            add_disgust = (500 - add_favorability) / cal_social
            target_data.status.setdefault(12, 0)
            target_data.status[12] += add_disgust
            character_handle.add_favorability(
                character_id, target_data.cid, add_favorability, target_change
            )


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_SMALL_SING_EXPERIENCE)
def handle_add_small_sing_experience(
    character_id: int, add_time: int, change_data: game_type.CharacterStatusChange
):
    """
    增加少量唱歌技能经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.knowledge.setdefault(15, 0)
    experience = 0.01 * add_time * character_data.knowledge_interest[15]
    character_data.knowledge[15] += experience
    change_data.knowledge.setdefault(15, 0)
    change_data.knowledge[15] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_SMALL_PLAY_MUSIC_EXPERIENCE)
def handle_add_small_play_music_experience(
    character_id: int, add_time: int, change_data: game_type.CharacterStatusChange
):
    """
    增加少量演奏技能经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.knowledge.setdefault(25, 0)
    experience = 0.01 * add_time * character_data.knowledge_interest[25]
    character_data.knowledge[25] += experience
    change_data.knowledge.setdefault(25, 0)
    change_data.knowledge[25] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_SMALL_ELOQUENCE_EXPERIENCE)
def handle_add_small_eloquence_experience(
    character_id: int, add_time: int, change_data: game_type.CharacterStatusChange
):
    """
    增加少量口才技能经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.knowledge.setdefault(12, 0)
    experience = 0.01 * add_time * character_data.knowledge_interest[12]
    character_data.knowledge[12] += experience
    change_data.knowledge.setdefault(12, 0)
    change_data.knowledge[12] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_SMALL_PERFORM_EXPERIENCE)
def handle_add_small_perform_experience(
    character_id: int, add_time: int, change_data: game_type.CharacterStatusChange
):
    """
    增加少量表演技能经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.knowledge.setdefault(11, 0)
    experience = 0.01 * add_time * character_data.knowledge_interest[11]
    character_data.knowledge[11] += experience
    change_data.knowledge.setdefault(11, 0)
    change_data.knowledge[11] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_SMALL_CEREMONY_EXPERIENCE)
def handle_add_small_ceremony_experience(
    character_id: int, add_time: int, change_data: game_type.CharacterStatusChange
):
    """
    增加少量礼仪技能经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.knowledge.setdefault(30, 0)
    experience = 0.01 * add_time * character_data.knowledge_interest[30]
    character_data.knowledge[30] += experience
    change_data.knowledge.setdefault(30, 0)
    change_data.knowledge[30] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_SMALL_SEX_EXPERIENCE)
def handle_add_small_sex_experience(
    character_id: int, add_time: int, change_data: game_type.CharacterStatusChange
):
    """
    增加少量性爱技能经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.knowledge.setdefault(9, 0)
    experience = 0.01 * add_time * character_data.knowledge_interest[9]
    character_data.knowledge[9] += experience
    change_data.knowledge.setdefault(9, 0)
    change_data.knowledge[9] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_SMALL_MOUTH_SEX_EXPERIENCE)
def handle_add_small_mouth_sex_experience(
    character_id: int, add_time: int, change_data: game_type.CharacterStatusChange
):
    """
    增加少量嘴部性爱技能经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    target_data.social_contact_data.setdefault(character_id, 0)
    if target_data.social_contact[character_id] >= 3:
        character_data.sex_experience.setdefault(0, 0)
        target_data.sex_experience.setdefault(0, 0)
        character_data.sex_experience[0] += add_time
        target_data.sex_experience[0] += add_time
        change_data.sex_experience.setdefault(0, 0)
        change_data.sex_experience[0] += add_time
        change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
        target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
        target_change.sex_experience.setdefault(0, 0)
        target_change.sex_experience[0] += add_time


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.FIRST_KISS)
def handle_first_kiss(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange):
    """
    记录初吻
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    target_data.social_contact_data.setdefault(character_id, 0)
    if target_data.social_contact[character_id] >= 3:
        if character_data.first_kiss == -1:
            character_data.first_kiss = target_data.cid
            if (not character_id) or (not target_data.cid):
                now_draw = draw.NormalDraw()
                now_draw.text = _("{character_name}失去了初吻\n").format(character_name=character_data.name)
                now_draw.width = window_width
                now_draw.draw()
        if target_data.first_kiss == -1:
            target_data.first_kiss = character_id
            if (not character_id) or (not target_data.cid):
                now_draw = draw.NormalDraw()
                now_draw.text = _("{character_name}失去了初吻\n").format(character_name=target_data.name)
                now_draw.width = window_width
                now_draw.draw()


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.FIRST_HAND_IN_HAND)
def handle_first_hand_in_hand(
    character_id: int, add_time: int, change_data: game_type.CharacterStatusChange
):
    """
    记录初次牵手
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    target_data.social_contact_data.setdefault(character_id, 0)
    if target_data.social_contact[character_id] >= 2:
        if character_data.first_hand_in_hand == -1:
            character_data.first_kiss = target_data.cid
        if target_data.first_hand_in_hand == -1:
            target_data.first_kiss = character_id


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_MEDIUM_HIT_POINT)
def handle_add_medium_hit_point(
    character_id: int, add_time: int, change_data: game_type.CharacterStatusChange
):
    """
    增加中量体力
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    """
    character_data: game_type.Character = cache.character_data[character_id]
    add_hit_point = add_time * 1.5
    character_data.hit_point += add_hit_point
    if character_data.hit_point > character_data.hit_point_max:
        add_hit_point -= character_data.hit_point - character_data.hit_point_max
        character_data.hit_point = character_data.hit_point_max
    change_data.hit_point += add_hit_point


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_MEDIUM_MANA_POINT)
def handle_add_medium_mana_point(
    character_id: int, add_time: int, change_data: game_type.CharacterStatusChange
):
    """
    增加中量气力
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    """
    character_data: game_type.Character = cache.character_data[character_id]
    add_mana_point = add_time * 3
    character_data.mana_point += add_mana_point
    if character_data.mana_point > character_data.mana_point_max:
        add_mana_point -= character_data.mana_point - character_data.mana_point_max
        character_data.mana_point = character_data.mana_point_max
    change_data.mana_point += add_mana_point
