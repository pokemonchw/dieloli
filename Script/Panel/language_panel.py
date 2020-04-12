from Script.Core import cache_contorl, text_loading, era_print
from Script.Design import attr_text


def see_character_language_panel(character_id: str):
    """
    查看角色语言能力面板
    Keyword arguments:
    character_id -- 角色Id
    """
    language_text_data = text_loading.get_game_data(text_loading.LANGUAGE_SKILLS_PATH)
    character_language = cache_contorl.character_data["character"][
        character_id
    ].language
    info_list = [
        language_text_data[language]["Name"]
        + ":"
        + attr_text.get_level_color_text(character_language[language])
        for language in character_language
    ]
    era_print.list_print(info_list, 4, "center")
