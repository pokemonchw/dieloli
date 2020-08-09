from Script.Core import (
    cache_contorl,
    text_loading,
    era_print,
    game_config,
    constant,
)
from Script.Design import (
    attr_print,
    attr_text,
    cmd_button_queue,
    attr_calculation,
)
from Script.Panel import (
    change_clothes_panel,
    use_item_panel,
    wear_item_panel,
    see_knowledge_panel,
    sex_experience_panel,
    language_panel,
    see_nature_panel,
    see_social_contact_panel,
)


def see_character_main_attr_panel(character_id: int):
    """
    查看角色主属性面板
    Keyword arguments:
    character_id -- 角色Id
    """
    title_1 = text_loading.get_text_data(
        constant.FilePath.STAGE_WORD_PATH, "1"
    )
    era_print.little_title_print(title_1)
    character_id_text = f"{text_loading.get_text_data(constant.FilePath.STAGE_WORD_PATH, '0')}{character_id}"
    character_data = cache_contorl.character_data[character_id]
    name = character_data.name
    nick_name = character_data.nick_name
    character_name = (
        text_loading.get_text_data(constant.FilePath.STAGE_WORD_PATH, "13")
        + name
    )
    character_nick_name = (
        text_loading.get_text_data(constant.FilePath.STAGE_WORD_PATH, "12")
        + nick_name
    )
    sex = character_data.sex
    sex_text = text_loading.get_text_data(
        constant.FilePath.STAGE_WORD_PATH, "2"
    ) + attr_text.get_sex_text(sex)
    name_text = (
        character_id_text
        + " "
        + character_name
        + " "
        + character_nick_name
        + " "
        + sex_text
    )
    hp_bar = attr_print.get_hp_or_mp_bar(
        character_id, "HitPoint", game_config.text_width / 2 - 4
    )
    era_print.list_print([name_text, hp_bar], 2, "center")
    era_print.line_feed_print()
    state_text = attr_text.get_state_text(character_id)
    mp_bar = attr_print.get_hp_or_mp_bar(
        character_id, "ManaPoint", game_config.text_width / 2 - 4
    )
    era_print.list_print([state_text, mp_bar], 2, "center")
    era_print.line_feed_print()
    era_print.little_line_print()
    stature_text = attr_text.get_stature_text(character_id)
    era_print.line_feed_print(stature_text)
    era_print.restart_line_print(".")
    era_print.list_print(
        [
            attr_text.get_character_dormitory_path_text(character_id),
            attr_text.get_character_classroom_path_text(character_id),
            attr_text.get_character_officeroom_path_text(character_id),
        ],
        3,
        "center",
    )
    era_print.little_line_print()
    character_species = f"{text_loading.get_text_data(constant.FilePath.STAGE_WORD_PATH, '15')}{character_data.species}"
    character_age = f"{text_loading.get_text_data(constant.FilePath.STAGE_WORD_PATH, '3')}{character_data.age}"
    birthday_text = f"{text_loading.get_text_data(constant.FilePath.STAGE_WORD_PATH,'140')}{character_data.birthday.month}{text_loading.get_text_data(constant.FilePath.STAGE_WORD_PATH,'60')}{character_data.birthday.day}{text_loading.get_text_data(constant.FilePath.STAGE_WORD_PATH,'61')}"
    era_print.list_print(
        [character_species, character_age, birthday_text], 3, "center"
    )
    era_print.restart_line_print(".")
    character_intimate = f"{text_loading.get_text_data(constant.FilePath.STAGE_WORD_PATH, '16')}{character_data.intimate}"
    character_graces = f"{text_loading.get_text_data(constant.FilePath.STAGE_WORD_PATH, '17')}{character_data.graces}"
    era_print.list_print([character_intimate, character_graces], 2, "center")
    era_print.restart_line_print(".")
    character_chest = character_data.chest["NowChest"]
    chest_group = attr_calculation.judge_chest_group(character_chest)
    chest_text = (
        text_loading.get_text_data(constant.FilePath.STAGE_WORD_PATH, "141")
        + text_loading.get_text_data(constant.FilePath.STAGE_WORD_PATH, "142")[
            chest_group
        ]
    )
    era_print.list_print([chest_text], 1, "center")
    era_print.restart_line_print(".")
    character_height = character_data.height["NowHeight"]
    character_weight = character_data.weight
    character_height_text = str(round(character_height, 2))
    character_weight_text = str(round(character_weight, 2))
    character_height_info = (
        text_loading.get_text_data(constant.FilePath.STAGE_WORD_PATH, "80")
        + character_height_text
    )
    character_weight_info = (
        text_loading.get_text_data(constant.FilePath.STAGE_WORD_PATH, "81")
        + character_weight_text
    )
    era_print.list_print(
        [character_height_info, character_weight_info], 2, "center"
    )
    era_print.restart_line_print(".")
    character_measurements = character_data.measurements
    character_bust = str(round(character_measurements["Bust"], 2))
    character_waist = str(round(character_measurements["Waist"], 2))
    character_hip = str(round(character_measurements["Hip"], 2))
    character_bust_info = (
        text_loading.get_text_data(constant.FilePath.STAGE_WORD_PATH, "82")
        + character_bust
    )
    character_waist_info = (
        text_loading.get_text_data(constant.FilePath.STAGE_WORD_PATH, "83")
        + character_waist
    )
    character_hip_info = (
        text_loading.get_text_data(constant.FilePath.STAGE_WORD_PATH, "84")
        + character_hip
    )
    era_print.list_print(
        [character_bust_info, character_waist_info, character_hip_info],
        3,
        "center",
    )
    era_print.restart_line_print(".")


def see_character_status_head_panel(character_id: str) -> str:
    """
    查看角色状态面板头部面板
    Keyword arguments:
    character_id -- 角色Id
    """
    era_print.little_title_print(
        text_loading.get_text_data(constant.FilePath.STAGE_WORD_PATH, "135")
    )
    era_print.line_feed_print(
        attr_text.get_see_attr_panel_head_character_info(character_id)
    )
    see_character_status_panel(character_id)


def see_character_status_panel(character_id: str):
    """
    查看角色状态面板
    Keyword arguments:
    character_id -- 角色Id
    """
    status_text_data = text_loading.get_text_data(
        constant.FilePath.STAGE_WORD_PATH, "134"
    )
    character_data = cache_contorl.character_data[character_id]
    status_data = character_data.status
    for state_type in status_data:
        era_print.son_title_print(status_text_data[state_type])
        now_status_data = status_data[state_type].copy()
        if state_type == "SexFeel":
            if character_data.sex == "Man":
                del now_status_data["VaginaDelight"]
                del now_status_data["ClitorisDelight"]
                del now_status_data["VaginaLubrication"]
            elif character_data.sex == "Woman":
                del now_status_data["PenisDelight"]
            elif character_data.sex == "Asexual":
                del now_status_data["VaginaDelight"]
                del now_status_data["ClitorisDelight"]
                del now_status_data["VaginaLubrication"]
                del now_status_data["PenisDelight"]
        now_status_text_list = [
            status_text_data[state]
            + ":"
            + str(round(now_status_data[state], 2))
            for state in now_status_data
        ]
        size = 7
        if len(now_status_text_list) < size:
            size = len(now_status_text_list)
        era_print.list_print(now_status_text_list, size, "center")
    era_print.line_feed_print()


def see_character_hp_and_mp_in_sence(character_id: int):
    """
    在场景中显示角色的HP和MP
    Keyword arguments:
    character_id -- 角色Id
    """
    if character_id == 0:
        attr_print.print_hp_and_mp_bar(character_id)
    else:
        character_id_text = (
            text_loading.get_text_data(constant.FilePath.STAGE_WORD_PATH, "0")
            + "0"
            + ":"
            + cache_contorl.character_data[0].name
        )
        target_id_text = (
            text_loading.get_text_data(constant.FilePath.STAGE_WORD_PATH, "0")
            + f"{character_id}"
            + ":"
            + cache_contorl.character_data[character_id].name
        )
        era_print.list_print([character_id_text, target_id_text], 2, "center")
        era_print.line_feed_print()
        player_bar = attr_print.get_hp_or_mp_bar(
            0, "HitPoint", game_config.text_width / 2 - 4
        )
        target_bar = attr_print.get_hp_or_mp_bar(
            character_id, "HitPoint", game_config.text_width / 2 - 4
        )
        era_print.list_print([player_bar, target_bar], 2, "center")
        era_print.line_feed_print()
        player_bar = attr_print.get_hp_or_mp_bar(
            0, "ManaPoint", game_config.text_width / 2 - 4
        )
        target_bar = attr_print.get_hp_or_mp_bar(
            character_id, "ManaPoint", game_config.text_width / 2 - 4
        )
        era_print.list_print([player_bar, target_bar], 2, "center")
        era_print.line_feed_print()


def see_character_equipment_panel(character_id: int):
    """
    查看角色装备面板
    Keyword arguments:
    character_id -- 角色Id
    """
    era_print.little_title_print(
        text_loading.get_text_data(constant.FilePath.STAGE_WORD_PATH, "37")
    )
    era_print.normal_print(
        attr_text.get_see_attr_panel_head_character_info(character_id)
    )
    change_clothes_panel.see_character_wear_clothes(character_id, False)


def see_character_item_panel(character_id: int):
    """
    查看角色道具面板
    Keyword arguments:
    character_id -- 角色Id
    """
    era_print.little_title_print(
        text_loading.get_text_data(constant.FilePath.STAGE_WORD_PATH, "38")
    )
    use_item_panel.see_character_item_panel(character_id)


def see_character_wear_item_panel(character_id: int):
    """
    查看角色穿戴道具面板
    Keyword arguments:
    character_id -- 角色Id
    """
    era_print.little_title_print(
        text_loading.get_text_data(constant.FilePath.STAGE_WORD_PATH, "40")
    )
    era_print.line_feed_print(
        attr_text.get_see_attr_panel_head_character_info(character_id)
    )
    wear_item_panel.see_character_wear_item_panel(character_id, False)


def see_character_knowledge_panel(character_id: int):
    """
    查看角色知识信息面板
    Keyword arguments:
    character_id -- 角色Id
    """
    era_print.little_title_print(
        text_loading.get_text_data(constant.FilePath.STAGE_WORD_PATH, "137")
    )
    era_print.line_feed_print(
        attr_text.get_see_attr_panel_head_character_info(character_id)
    )
    see_knowledge_panel.see_character_knowledge_panel(character_id)


def see_character_sex_experience_panel(character_id: int):
    """
    查看角色性经验面板
    Keyword arguments:
    character_id -- 角色Id
    """
    era_print.little_title_print(
        text_loading.get_text_data(constant.FilePath.STAGE_WORD_PATH, "138")
    )
    era_print.line_feed_print(
        attr_text.get_see_attr_panel_head_character_info(character_id)
    )
    sex_experience_panel.see_character_sex_experience_panel(character_id)


def see_character_language_panel(character_id: int):
    """
    查看角色语言能力面板
    Keyword arguments:
    character_id -- 角色Id
    """
    era_print.little_title_print(
        text_loading.get_text_data(constant.FilePath.STAGE_WORD_PATH, "139")
    )
    era_print.line_feed_print(
        attr_text.get_see_attr_panel_head_character_info(character_id)
    )
    language_panel.see_character_language_panel(character_id)


def see_character_nature_panel(character_id: int):
    """
    查看角色性格面板
    Keyword arguments:
    character_id -- 角色Id
    """
    era_print.little_title_print(
        text_loading.get_text_data(constant.FilePath.STAGE_WORD_PATH, "139")
    )
    era_print.line_feed_print(
        attr_text.get_see_attr_panel_head_character_info(character_id)
    )
    see_nature_panel.see_character_nature_panel(character_id)


def see_character_social_contact_panel(character_id: int):
    """
    查看角色社交面板
    Keyword arguments:
    character_id -- 角色Id
    """
    era_print.little_title_print(
        text_loading.get_text_data(constant.FilePath.STAGE_WORD_PATH, "145")
    )
    era_print.line_feed_print(
        attr_text.get_see_attr_panel_head_character_info(character_id)
    )
    see_social_contact_panel.see_character_social_contact_panel(character_id)


def ask_for_see_attr() -> list:
    """
    查看角色属性时输入处理面板
    """
    era_print.restart_line_print()
    ask_data = text_loading.get_text_data(
        constant.FilePath.CMD_PATH, constant.CmdMenu.SEE_ATTR_PANEL_HANDLE
    ).copy()
    now_panel_id = cache_contorl.panel_state["AttrShowHandlePanel"]
    ask_list = list(ask_data.values())
    cmd_button_queue.option_str(
        None,
        5,
        "center",
        False,
        False,
        ask_list,
        now_panel_id,
        list(ask_data.keys()),
    )
    del ask_data[now_panel_id]
    return list(ask_data.keys())


def ask_for_see_attr_cmd() -> list:
    """
    查看属性页显示控制
    """
    era_print.restart_line_print("~")
    yrn = cmd_button_queue.option_int(
        constant.CmdMenu.SEE_ATTR_ON_EVERY_TIME,
        3,
        cmd_size="center",
        askfor=False,
    )
    return yrn


def input_attr_over_panel():
    """
    创建角色完成时确认角色属性输入处理面板
    """
    yrn = cmd_button_queue.option_int(
        constant.CmdMenu.ACKNOWLEDGEMENT_ATTRIBUTE, askfor=False
    )
    return yrn


panel_data = {
    "MainAttr": see_character_main_attr_panel,
    "Equipment": see_character_equipment_panel,
    "Status": see_character_status_head_panel,
    "Item": see_character_item_panel,
    "WearItem": see_character_wear_item_panel,
    "SexExperience": see_character_sex_experience_panel,
    "Knowledge": see_character_knowledge_panel,
    "Language": see_character_language_panel,
    "Nature": see_character_nature_panel,
    "SocialContact": see_character_social_contact_panel,
}
