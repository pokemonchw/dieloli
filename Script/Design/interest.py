import random
import numpy
from Script.Core import cache_contorl, text_loading


def init_character_interest():
    """
    初始化全部角色兴趣/精力/天赋数值分配
    """
    interest_list = []
    language_skills = text_loading.get_game_data(text_loading.LANGUAGE_SKILLS_PATH)
    interest_list += list(language_skills.keys())
    knowledge_data = text_loading.get_game_data(text_loading.KNOWLEDGE_PATH)
    for knowledge_tag in knowledge_data:
        knowledge_list = knowledge_data[knowledge_tag]
        interest_list += list(knowledge_list["Knowledge"].keys())
    interest_average = 100 / len(interest_list)
    for character in cache_contorl.character_data["character"]:
        now_interest_value_max = 100
        now_interest_list = interest_list.copy()
        numpy.random.shuffle(now_interest_list)
        for interest in now_interest_list:
            if interest != now_interest_list[-1]:
                now_interest_average = now_interest_value_max / len(now_interest_list)
                now_inter_value = now_interest_average * random.uniform(0.75, 1.25)
                now_interest_value_max -= now_inter_value
                cache_contorl.character_data["character"][character].interest[
                    interest
                ] = (now_inter_value / interest_average)
            else:
                cache_contorl.character_data["character"][character].interest[
                    interest
                ] = now_interest_value_max
