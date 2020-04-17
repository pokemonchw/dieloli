from Script.Core import game_config, game_data, game_path_config, constant

game_path = game_path_config.game_path


LANGUAGE = game_config.language


def get_text_data(text_path_id: str, text_id: str) -> str:
    """
    按文件id和文本id读取指定文本数据
    Keyword arguments:
    text_path_id -- 文件id
    text_id -- 文件下的文本id
    """
    if text_path_id in ["FontConfig", "BarConfig"]:
        return game_data.game_data[text_path_id][text_id]
    else:
        return game_data.game_data[LANGUAGE][text_path_id][text_id]


def get_game_data(text_path_id: str) -> dict:
    """
    按文件id读取文件数据
    Keyword arguments:
    text_path_id -- 文件id
    """
    if text_path_id in [
        constant.FilePath.FONT_CONFIG_PATH,
        constant.FilePath.BAR_CONFIG_PATH,
    ]:
        return game_data.game_data[text_path_id]
    else:
        return game_data.game_data[LANGUAGE][text_path_id]


def get_character_data(character_name: str) -> dict:
    """
    按角色名获取预设的角色模板数据
    Keyword arguments:
    character_name -- 角色名
    """
    return game_data.game_data[LANGUAGE]["character"][character_name]
