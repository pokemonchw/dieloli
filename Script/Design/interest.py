import random
import numpy
from Script.Core import cache_control, constant, game_type
from Script.Config import game_config

cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """


def init_character_interest():
    """
    初始化全部角色兴趣/精力/天赋数值分配
    """
    interest_list = []
    language_skills = list(game_config.config_language.keys())
    language_average = 100 / len(language_skills)
    knowledge_skills = list(game_config.config_knowledge.keys())
    knowledge_average = 100 / len(knowledge_skills)
    for character in cache.character_data:
        now_knowledge_value_max = 100
        now_language_value_max = 100
        numpy.random.shuffle(knowledge_skills)
        numpy.random.shuffle(language_skills)
        for knowledge in knowledge_skills:
            if knowledge != knowledge_skills[-1]:
                now_interest_average = now_knowledge_value_max / len(knowledge_skills)
                now_inter_value = now_interest_average * random.uniform(0.75, 1.25)
                now_knowledge_value_max -= now_inter_value
                cache.character_data[character].knowledge_interest[knowledge] = (
                    now_inter_value / knowledge_average
                )
            else:
                cache.character_data[character].knowledge_interest[knowledge] = (
                    now_knowledge_value_max / knowledge_average
                )
        for language in language_skills:
            if language != language_skills[-1]:
                now_interest_average = now_language_value_max / len(language_skills)
                now_inter_value = now_interest_average * random.uniform(0.75, 1.25)
                now_language_value_max -= now_inter_value
                cache.character_data[character].language_interest[language] = (
                    now_inter_value / language_average
                )
            else:
                cache.character_data[character].language_interest[language] = (
                    now_language_value_max / language_average
                )
