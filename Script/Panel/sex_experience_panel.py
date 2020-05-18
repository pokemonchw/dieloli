from Script.Core import cache_contorl, text_loading, era_print, constant
from Script.Design import attr_text


def see_character_sex_experience_panel(character_id: int):
    """
    查看角色性经验面板
    Keyword arguments:
    character_id -- 角色Id
    """
    era_print.little_line_print()
    era_print.line_feed_print(
        text_loading.get_text_data(constant.FilePath.STAGE_WORD_PATH, "5")
    )
    character_data = cache_contorl.character_data[character_id]
    character_sex_grade_list = character_data.sex_grade
    character_sex = cache_contorl.character_data[character_id].sex
    character_sex_grade_text_list = attr_text.get_sex_grade_text_list(
        character_sex_grade_list, character_sex
    )
    era_print.list_print(character_sex_grade_text_list, 4, "center")
    era_print.line_feed_print(
        text_loading.get_text_data(constant.FilePath.STAGE_WORD_PATH, "7")
    )
    character_engraving = character_data.engraving
    character_engraving_text = attr_text.get_engraving_text(
        character_engraving
    )
    era_print.list_print(character_engraving_text, 3, "center")
