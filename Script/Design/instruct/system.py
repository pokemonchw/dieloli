from types import FunctionType
from Script.Core import cache_control, game_type, get_text
from Script.Design import constant, handle_instruct
from Script.UI.Panel import see_character_info_panel, see_save_info_panel
from Script.Config import normal_config
from Script.UI.Moudle import draw


cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """
_: FunctionType = get_text._
""" 翻译api """
width: int = normal_config.config_normal.text_width
""" 屏幕宽度 """


@handle_instruct.add_instruct(
    constant.Instruct.SEE_ATTR,
    constant.InstructType.SYSTEM,
    _("查看属性"),
    {constant.Premise.HAVE_TARGET},
)
def handle_see_attr():
    """查看属性"""
    see_character_info_panel.line_feed.draw()
    now_draw = see_character_info_panel.SeeCharacterInfoInScenePanel(
        cache.character_data[0].target_character_id, width
    )
    now_draw.draw()


@handle_instruct.add_instruct(constant.Instruct.SEE_OWNER_ATTR, constant.InstructType.SYSTEM, _("查看自身属性"), {})
def handle_see_owner_attr():
    """查看自身属性"""
    see_character_info_panel.line_feed.draw()
    now_draw = see_character_info_panel.SeeCharacterInfoInScenePanel(0, width)
    now_draw.draw()


@handle_instruct.add_instruct(
    constant.Instruct.VIEW_CHARACTER_STATUS_LIST,
    constant.InstructType.SYSTEM,
    _("Alpha监控台"),
    {
        constant.Premise.IN_STUDENT_UNION_OFFICE,
    },
)
def handle_view_character_status_list():
    """打开角色状态监控面板指令"""
    cache.now_panel_id = constant.Panel.VIEW_CHARACTER_STATUS_LIST


@handle_instruct.add_instruct(
    constant.Instruct.COLLECTION_CHARACTER,
    constant.InstructType.SYSTEM,
    _("收藏角色"),
    {constant.Premise.TARGET_IS_NOT_COLLECTION, constant.Premise.TARGET_NO_PLAYER},
)
def handle_collection_character():
    """处理收藏角色指令"""
    character_data: game_type.Character = cache.character_data[0]
    target_character_id = character_data.target_character_id
    if target_character_id not in character_data.collection_character:
        character_data.collection_character.add(target_character_id)


@handle_instruct.add_instruct(
    constant.Instruct.UN_COLLECTION_CHARACTER,
    constant.InstructType.SYSTEM,
    _("取消收藏"),
    {constant.Premise.TARGET_IS_COLLECTION, constant.Premise.TARGET_NO_PLAYER},
)
def handle_un_collection_character():
    """处理取消指令"""
    character_data: game_type.Character = cache.character_data[0]
    target_character_id = character_data.target_character_id
    if target_character_id in character_data.collection_character:
        character_data.collection_character.remove(target_character_id)


@handle_instruct.add_instruct(
    constant.Instruct.COLLECTION_SYSTEM,
    constant.InstructType.SYSTEM,
    _("启用收藏模式"),
    {constant.Premise.UN_COLLECTION_SYSTEM},
)
def handle_collection_system():
    """处理启用收藏模式指令"""
    cache.is_collection = 1
    now_draw = draw.WaitDraw()
    now_draw.width = width
    now_draw.text = _("\n现在只会显示被收藏的角色的信息了！\n")
    now_draw.draw()


@handle_instruct.add_instruct(
    constant.Instruct.UN_COLLECTION_SYSTEM,
    constant.InstructType.SYSTEM,
    _("关闭收藏模式"),
    {constant.Premise.IS_COLLECTION_SYSTEM},
)
def handle_un_collection_system():
    """处理关闭收藏模式指令"""
    cache.is_collection = 0
    now_draw = draw.WaitDraw()
    now_draw.width = width
    now_draw.text = _("\n现在会显示所有角色的信息了！\n")
    now_draw.draw()


@handle_instruct.add_instruct(constant.Instruct.SAVE, constant.InstructType.SYSTEM, _("读写存档"), {})
def handle_save():
    """处理读写存档指令"""
    now_panel = see_save_info_panel.SeeSaveListPanel(width, 1)
    now_panel.draw()
