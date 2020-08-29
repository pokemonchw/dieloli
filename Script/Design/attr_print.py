from Script.Core import (
    text_loading,
    era_print,
    cache_contorl,
    game_config,
    constant,
)
from Script.Design import proportional_bar

point_text_data = {
    "HitPoint": text_loading.get_text_data(constant.FilePath.STAGE_WORD_PATH, "8"),
    "ManaPoint": text_loading.get_text_data(constant.FilePath.STAGE_WORD_PATH, "9"),
}


def print_hp_and_mp_bar(character_id: str):
    """
    绘制角色的hp和mp(有比例图)，自动居中处理，结尾换行
    Keyword arguments:
    character_id -- 角色id
    """
    hp_bar = get_hp_or_mp_bar(character_id, "HitPoint", game_config.text_width / 2 - 4)
    mp_bar = get_hp_or_mp_bar(character_id, "ManaPoint", game_config.text_width / 2 - 4)
    era_print.line_feed_print()
    era_print.list_print([hp_bar, mp_bar], 2, "center")
    era_print.line_feed_print()


def get_hp_or_mp_bar(character_id: str, bar_id: str, text_width: int):
    """
    获取角色的hp或mp比例图，按给定宽度居中
    Keyword arguments:
    character_id -- 角色Id
    bar_id -- 绘制的比例条类型(hp/mp)
    text_width -- 比例条总宽度
    """
    character_data = cache_contorl.character_data[character_id]
    if bar_id == "HitPoint":
        character_point = character_data.hit_point
        character_max_point = character_data.hit_point_max
    else:
        character_point = character_data.mana_point
        character_max_point = character_data.mana_point_max
    point_text = point_text_data[bar_id]
    return proportional_bar.get_proportional_bar(
        point_text, character_max_point, character_point, bar_id + "bar", text_width,
    )


def get_hp_and_mp_text(character_id: str) -> str:
    """
    获取角色的hp和mp文本
    Keyword arguments:
    character_id -- 角色id
    """
    character_data = cache_contorl.character_data[character_id]
    character_hit_point = character_data.hit_point
    character_max_hit_point = character_data.hit_point_max
    hit_point_text = text_loading.get_text_data(constant.FilePath.STAGE_WORD_PATH, "8")
    hp_text = hit_point_text + "(" + str(character_hit_point) + "/" + str(character_max_hit_point) + ")"
    character_mana_point = character_data.mana_point
    character_max_mana_point = character_data.mana_point_max
    mana_point_text = text_loading.get_text_data(constant.FilePath.STAGE_WORD_PATH, "9")
    mp_text = mana_point_text + "(" + str(character_mana_point) + "/" + str(character_max_mana_point) + ")"
    return hp_text + " " + mp_text
