import os
from Script.Core import cache_contorl, game_path_config, game_config
from Script.Behavior import student, teacher

game_path = game_path_config.game_path
language = game_config.language
character_list_path = os.path.join(game_path, "data", language, "character")

behavior_tem_data = {"Student": student, "Teacher": teacher}


def init_character_behavior():
    """
    角色行为树总控制
    """
    for npc in cache_contorl.character_data:
        if npc == 0:
            continue
        character_occupation_judge(npc)


def character_occupation_judge(character_id: int):
    """
    判断角色职业并指定对应行为树
    Keyword arguments:
    character_id -- 角色id
    """
    character_data = cache_contorl.character_data[character_id]
    template = behavior_tem_data[character_data.occupation]
    template.behavior_init(character_id)
