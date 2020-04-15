from Script.Core import cache_contorl, text_loading, era_print
from Script.Design import cmd_button_queue


def see_character_nature_panel(character_id: str):
    """
    用于菜单中查看角色性格信息面板
    Keyword arguments:
    character_id -- 角色Id
    """
    see_character(character_id, False)


def see_character_nature_change_panel(character_id: str) -> list:
    """
    用于查看和切换角色性格信息面板
    Keyword arguments:
    character_id -- 角色Id
    Return arguments:
    list -- 按钮列表
    """
    return see_character(character_id, True)


def see_character(character_id: str, judge: bool) -> list:
    """
    用于任何时候查看角色性格信息面板
    Keyword arguments:
    character_id -- 角色Id
    judge -- 绘制按钮校验
    Return arguments:
    list -- 按钮列表
    """
    nature_text_data = text_loading.get_game_data(text_loading.NATURE_PATH)
    character_nature = cache_contorl.character_data["character"][
        character_id
    ].nature
    cmd_list = []
    for nature in nature_text_data:
        nature_text = nature_text_data[nature]["Name"]
        if "Good" in nature_text:
            now_nature_values = [
                character_nature[son_nature]
                for son_nature in nature_text_data[nature]["Factor"]
            ]
            now_nature_value = sum(now_nature_values)
            now_nature_max = len(now_nature_values) * 100
            if now_nature_value < now_nature_max / 2:
                nature_text = nature_text["Bad"]
            else:
                nature_text = nature_text["Good"]
        era_print.son_title_print(nature_text)
        info_list = [
            nature_text_data[nature]["Factor"][son_nature][
                judge_nature_good(character_nature[son_nature])
            ]
            for son_nature in nature_text_data[nature]["Factor"]
        ]
        if judge:
            now_son_list = [son for son in nature_text_data[nature]["Factor"]]
            cmd_list += now_son_list
            cmd_button_queue.option_str(
                None,
                len(now_son_list),
                "center",
                False,
                False,
                info_list,
                "",
                now_son_list,
            )
        else:
            era_print.list_print(info_list, len(info_list), "center")
    return cmd_list


def judge_nature_good(nature: int) -> str:
    """
    校验性格倾向
    Keyword arguments:
    nature -- 性格数值
    Return arguments:
    str -- 好坏
    """
    if nature < 50:
        return "Bad"
    return "Good"
