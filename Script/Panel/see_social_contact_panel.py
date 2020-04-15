from Script.Core import cache_contorl, text_loading, era_print


def see_character_social_contact_panel(character_id: int):
    """
    查看角色社交信息面板
    Keyword arguments:
    character_id -- 角色Id
    """
    social_contact_text_data = text_loading.get_text_data(
        text_loading.STAGE_WORD_PATH, "144"
    )
    character_social_contact = cache_contorl.character_data["character"][
        character_id
    ].social_contact
    for social in social_contact_text_data:
        era_print.son_title_print(social_contact_text_data[social])
        if character_social_contact[social] == {}:
            era_print.normal_print(
                text_loading.get_text_data(text_loading.MESSAGE_PATH, "40")
            )
        else:
            size = 10
            if len(character_social_contact[social]) < 10:
                size = len(character_social_contact[social])
            name_list = [
                cache_contorl.character_data["character"][character_id].name
                for character_id in character_social_contact[social]
            ]
            era_print.list_print(name_list, size, "center")
