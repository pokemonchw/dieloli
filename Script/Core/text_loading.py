from Script.Core import game_config, game_data, game_path_config

game_path = game_path_config.game_path

MENU_RESTART = "1"
MENU_QUIT = "2"
MENU_SETTING = "3"
MENU_ABBOUT = "4"
MENU_FILE = "5"
MENU_OTHER = "6"

LANGUAGE = game_config.language

MESSAGE_PATH = "MessageList"
CMD_PATH = "CmdText"
MENU_PATH = "MenuText"
ROLE_PATH = "RoleAttributes"
STAGE_WORD_PATH = "StageWord"
ERROR_PATH = "ErrorText"
ATTR_TEMPLATE_PATH = "AttrTemplate"
SYSTEM_TEXT_PATH = "SystemText"
NAME_LIST_PATH = "NameIndex"
FAMILY_NAME_LIST_PATH = "FamilyIndex"
FONT_CONFIG_PATH = "FontConfig"
BAR_CONFIG_PATH = "BarConfig"
PHASE_COURSE_PATH = "PhaseCourse"
COURSE_PATH = "Course"
COURSE_SESSION_PATH = "CourseSession"
KNOWLEDGE_PATH = "Knowledge"
LANGUAGE_SKILLS_PATH = "LanguageSkills"
EQUIPMENT_PATH = "Equipment"
STATURE_DESCRIPTION_PATH = "StatureDescription"
CHARACTER_STATE_PATH = "CharacterState"
WEAR_ITEM_PATH = "WearItem"
NATURE_PATH = "Nature"


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
    if text_path_id in [FONT_CONFIG_PATH, BAR_CONFIG_PATH]:
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
