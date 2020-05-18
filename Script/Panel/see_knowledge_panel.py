from Script.Core import cache_contorl, text_loading, era_print, constant
from Script.Design import attr_text


def see_character_knowledge_panel(character_id: str):
    """
    查看角色技能信息面板
    Keyword arguments:
    character_id -- 角色Id
    """
    knowledge_text_data = text_loading.get_game_data(
        constant.FilePath.KNOWLEDGE_PATH
    )
    character_knowledge = cache_contorl.character_data[character_id].knowledge
    for knowledge in knowledge_text_data:
        era_print.son_title_print(knowledge_text_data[knowledge]["Name"])
        if knowledge in character_knowledge:
            info_list = [
                knowledge_text_data[knowledge]["Knowledge"][skill]["Name"]
                + ":"
                + attr_text.get_level_color_text(
                    character_knowledge[knowledge][skill]
                )
                for skill in character_knowledge[knowledge]
            ]
            era_print.list_print(info_list, 6, "center")
