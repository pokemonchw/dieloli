import os
from functools import wraps
from Script.Core import cache_contorl, game_path_config, game_config

game_path = game_path_config.game_path
language = game_config.language
character_list_path = os.path.join(game_path, "data", language, "character")


def init_character_behavior():
    """
    角色行为树总控制
    """
    while 1:
        if (
            len(cache_contorl.over_behavior_character)
            == len(cache_contorl.character_data) - 1
        ):
            break
        for npc in cache_contorl.character_data:
            if npc == 0 or npc in cache_contorl.over_behavior_character:
                continue
            character_occupation_judge(npc)
    cache_contorl.over_behavior_character = {}


def add_behavior(occupation: str, status: int):
    """
    添加角色行为控制器
    Keyword arguments:
    occupation -- 职业
    status -- 状态id
    """

    def decoraror(func):
        @wraps(func)
        def return_wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        cache_contorl.behavior_tem_data.setdefault(occupation, {})
        cache_contorl.behavior_tem_data[occupation][status] = return_wrapper
        return return_wrapper

    return decoraror


def character_occupation_judge(character_id: int):
    """
    判断角色职业并指定对应行为树
    Keyword arguments:
    character_id -- 角色id
    """
    character_data = cache_contorl.character_data[character_id]
    occupation = character_data.occupation
    if occupation not in cache_contorl.behavior_tem_data:
        if character_data.age > 18:
            occupation = "Teacher"
        else:
            occupation = "Student"
    if character_data.state not in cache_contorl.behavior_tem_data[occupation]:
        occupation = "Default"
    if cache_contorl.behavior_tem_data[occupation][character_data.state](
        character_id
    ):
        cache_contorl.over_behavior_character[character_id] = 0
