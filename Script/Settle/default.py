import datetime
from typing import Dict
from Script.Design import (
    settle_behavior,
    talk,
    game_time,
    map_handle,
    character,
    attr_calculation,
)
from Script.Core import constant, cache_control, game_type
from Script.Config import game_config

cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """


@settle_behavior.add_settle_behavior(constant.Behavior.REST)
def settle_rest(character_id: int, now_time: datetime.datetime) -> game_type.CharacterStatusChange:
    """
    结算角色休息行为
    Keyword arguments:
    character_id -- 角色id
    now_time -- 结算时间
    Return arguments:
    game_type.CharacterStatusChange -- 行为改变的角色状态
    """
    character_data: game_type.Character = cache.character_data[character_id]
    start_time = character_data.behavior.start_time
    add_time = int((now_time - start_time).seconds / 60)
    add_hit_point = add_time * 5
    add_mana_point = add_time * 10
    character_data.hit_point += add_hit_point
    character_data.mana_point += add_mana_point
    now_change_data = game_type.CharacterStatusChange()
    if character_data.hit_point > character_data.hit_point_max:
        add_hit_point -= character_data.hit_point - character_data.hit_point_max
        character_data.hit_point = character_data.hit_point_max
    if character_data.mana_point > character_data.mana_point_max:
        add_mana_point -= character_data.mana_point - character_data.mana_point_max
        character_data.mana_point = character_data.mana_point_max
    now_change_data.hit_point = add_hit_point
    now_change_data.mana_point = add_mana_point
    if character_data.target_character_id != character_id:
        add_favorability = character.calculation_favorability(
            character_id, character_data.target_character_id, add_time
        )
        target_data: game_type.Character = cache.character_data[character_data.target_character_id]
        target_data.favorability.setdefault(character_id, 0)
        target_data.favorability[character_id] += add_favorability
        now_change_data.favorability[target_data.cid] = add_favorability
    return now_change_data


@settle_behavior.add_settle_behavior(constant.Behavior.MOVE)
def settle_move(character_id: int, now_time: datetime.datetime) -> game_type.CharacterStatusChange:
    """
    结算角色移动行为
    Keyword arguments:
    character_id -- 角色id
    now_time -- 结算时间
    Return arguments:
    game_type.CharacterStatusChange -- 行为改变的角色状态
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.target_character_id = character_id
    now_change_data = game_type.CharacterStatusChange()
    start_time = character_data.behavior.start_time
    add_time = int((now_time - start_time).seconds / 60)
    if character_data.mana_point >= add_time:
        character_data.mana_point -= add_time
    else:
        add_time -= character_data.mana_point
        now_change_data.mana_point -= character_data.mana_point
        character_data.mana_point = 0
        character_data.hit_point -= add_time * 10
    map_handle.character_move_scene(
        character_data.position,
        character_data.behavior.move_target,
        character_id,
    )
    return now_change_data


@settle_behavior.add_settle_behavior(constant.Behavior.EAT)
def settle_eat(character_id: int, now_time: datetime.datetime) -> game_type.CharacterStatusChange:
    """
    结算角色进食行为
    Keyword arguments:
    character_id -- 角色id
    now_time -- 结算时间
    Return arguments:
    game_type.CharacterStatusChange -- 行为改变的角色状态
    """
    character_data = cache.character_data[character_id]
    now_change_data = game_type.CharacterStatusChange()
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
            now_change_data.status.setdefault(feel, 0)
            if feel in {27, 28}:
                return_data[feel] -= now_feel_value
                character_data.status[feel] -= now_feel_value
                if character_data.status[feel] < 0:
                    character_data.status[feel] = 0
                now_change_data.status[feel] -= now_feel_value
            else:
                character_data.status[feel] += now_feel_value
                return_data[feel] += now_feel_value
                now_change_data.status[feel] += now_feel_value
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
    return return_data


@settle_behavior.add_settle_behavior(constant.Behavior.CHAT)
def settle_chat(character_id: int, now_time: datetime.datetime) -> game_type.CharacterStatusChange:
    """
    结算角色闲聊行为
    Keyword arguments:
    character_id -- 角色id
    now_time -- 结算时间
    Return arguments:
    game_type.CharacterStatusChange -- 行为改变的角色状态
    """
    character_data: game_type.Character = cache.character_data[character_id]
    now_change_data = game_type.CharacterStatusChange()
    if character_data.target_character_id != character_id:
        start_time = character_data.behavior.start_time
        add_time = int((now_time - start_time).seconds / 60)
        add_favorability = character.calculation_favorability(
            character_id, character_data.target_character_id, add_time * 1.5
        )
        target_data: game_type.Character = cache.character_data[character_data.target_character_id]
        target_data.favorability.setdefault(character_id, 0)
        target_data.favorability[character_id] += add_favorability
        now_change_data.favorability[character_data.target_character_id] = add_favorability
    return now_change_data


def settle_social_contact(
    character_id: int, knowledge: int, now_time: datetime.datetime
) -> game_type.CharacterStatusChange:
    """
    结算角色社交技能行为
    Keyword arguments:
    character_id -- 角色id
    knowledge -- 技能id
    now_time -- 结算时间
    Return arguments:
    game_type.CharacterStatusChange -- 行为改变的角色状态
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact_data
    now_change_data = game_type.CharacterStatusChange()
    start_time = character_data.behavior.start_time
    add_time = int((now_time - start_time).seconds / 60)
    if character_data.mana_point >= add_time:
        character_data.mana_point -= add_time
        now_change_data.mana_point -= add_time
    else:
        now_add_time = add_time
        now_change_data.mana_point -= character_data.mana_point
        now_add_time -= character_data.mana_point
        character_data.mana_point = 0
        now_add_time *= 10
        now_change_data.hit_point -= now_add_time
        now_change_data.hit_point -= now_add_time
    now_experience = add_time * character_data.knowledge_interest[knowledge]
    character_data.knowledge.setdefault(knowledge, 0)
    character_data.knowledge[knowledge] += now_experience
    now_change_data.knowledge[knowledge] = now_experience
    if character_data.target_character_id != character_id:
        target_data: game_type.Character = cache.character_data[character_data.target_character_id]
        now_level = attr_calculation.get_experience_level_weight(character_data.knowledge[knowledge])
        add_favorability = now_level * add_time + add_time * character_data.knowledge_interest[knowledge]
        target_data.favorability.setdefault(character_id, 0)
        target_data.favorability[character_id] += add_favorability
        now_change_data.favorability[target_data.cid] = add_favorability
    return now_change_data


@settle_behavior.add_settle_behavior(constant.Behavior.PLAY_PIANO)
def settle_play_piano(character_id: int, now_time: datetime.datetime) -> game_type.CharacterStatusChange:
    """
    结算角色弹奏钢琴行为
    Keyword arguments:
    character_id -- 角色id
    now_time -- 结算时间
    Return arguments:
    game_type.CharacterStatusChange -- 行为改变的角色状态
    """
    return settle_social_contact(character_id, 25, now_time)


@settle_behavior.add_settle_behavior(constant.Behavior.SINGING)
def settle_singing(character_id: int, now_time: datetime.datetime) -> game_type.CharacterStatusChange:
    """
    结算角色唱歌行为
    Keyword arguments:
    character_id -- 角色id
    now_time -- 结算时间
    Return arguments:
    game_type.CharacterStatusChange -- 行为改变的角色状态
    """
    return settle_social_contact(character_id, 15, now_time)


@settle_behavior.add_settle_behavior(constant.Behavior.TOUCH_HEAD)
def settle_touch_head(character_id: int, now_time: datetime.datetime) -> game_type.CharacterStatusChange:
    """
    结算角色摸头行为
    Keyword arguments:
    character_id -- 角色id
    now_time -- 结算时间
    Return arguments:
    game_type.CharacterStatusChange -- 行为改变的角色状态
    """
    character_data: game_type.Character = cache.character_data[character_id]
    now_change_data = game_type.CharacterStatusChange()
    if character_data.target_character_id != character_id:
        start_time = character_data.behavior.start_time
        add_time = int((now_time - start_time).seconds / 60)
        add_favorability = character.calculation_favorability(
            character_id, character_data.target_character_id, add_time * 1.5
        )
        target_data: game_type.Character = cache.character_data[character_data.target_character_id]
        target_data.favorability.setdefault(character_id, 0)
        add_favorability_coefficient = add_favorability / (add_time * 1.5)
        social = 0
        if character_id in target_data.social_contact_data:
            social = target_data.social_contact_data[character_id]
        if social >= 3:
            add_favorability += add_favorability_coefficient * social
        else:
            add_favorability -= add_favorability_coefficient * social
        target_data.favorability[character_id] += add_favorability
        now_change_data.favorability[character_data.target_character_id] = add_favorability
    return now_change_data
