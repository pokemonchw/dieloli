from types import FunctionType
from Script.Core import cache_control, game_type, get_text, py_cmd
from Script.Design import constant, handle_instruct
from Script.UI.Panel import see_character_info_panel, see_save_info_panel, achieve_panel
from Script.Config import normal_config
from Script.UI.Model import draw, panel


cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """
_: FunctionType = get_text._
""" 翻译api """
width: int = normal_config.config_normal.text_width
""" 屏幕宽度 """
line_feed_draw = draw.NormalDraw()
""" 绘制换行对象 """
line_feed_draw.text = "\n"


@handle_instruct.add_instruct(
    constant.Instruct.SEE_ATTR,
    constant.InstructType.SYSTEM,
    _("查看属性"),
    {constant.Premise.HAVE_TARGET},
)
def handle_see_attr():
    """查看属性"""
    py_cmd.clr_cmd()
    now_draw = see_character_info_panel.SeeCharacterInfoInScenePanel(
        cache.character_data[0].target_character_id, width
    )
    now_draw.draw()


@handle_instruct.add_instruct(constant.Instruct.SEE_OWNER_ATTR, constant.InstructType.SYSTEM, _("查看自身属性"), {})
def handle_see_owner_attr():
    """查看自身属性"""
    py_cmd.clr_cmd()
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
    constant.Instruct.VIEW_CLUB_LIST,
    constant.InstructType.SYSTEM,
    _("申请加入社团"),
    {
        constant.Premise.IN_STUDENT_UNION_OFFICE,
        constant.Premise.NOT_JOINED_CLUB
    },
)
def handle_see_club_list():
    """ 打开查看社团列表的指令 """
    cache.now_panel_id = constant.Panel.VIEW_CLUB_LIST


@handle_instruct.add_instruct(
    constant.Instruct.VIEW_CLUB_INFO,
    constant.InstructType.SYSTEM,
    _("查看社团信息"),
    {
        constant.Premise.IS_JOINED_CLUB
    },
)
def handle_see_club_info():
    cache.now_panel_id = constant.Panel.VIEW_CLUB_INFO


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


@handle_instruct.add_instruct(
    constant.Instruct.SET_NICK_NAME,
    constant.InstructType.SYSTEM,
    _("设置昵称"),
    {constant.Premise.HAVE_TARGET},
)
def handle_set_nickname():
    """处理设置昵称指令"""
    if 1:
        return
    character_data: game_type.Character = cache.character_data[0]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    ask_name_panel = panel.AskForOneMessage()
    ask_name_panel.set(_("想要怎么称呼{target_name}?").format(target_name=target_data.name), 10)
    ask_name_panel.donot_return_null_str = False
    not_num_error = draw.NormalDraw()
    not_num_error.text = _("角色名不能为纯数字，请重新输入\n")
    not_system_error = draw.NormalDraw()
    not_system_error.text = _("角色名不能为系统保留字，请重新输入\n")
    not_name_error = draw.NormalDraw()
    not_name_error.text = _("已有角色使用该姓名，请重新输入\n")
    line_feed_draw.draw()
    while 1:
        now_name = ask_name_panel.draw()
        if now_name.isdigit():
            not_num_error.draw()
            continue
        if now_name in get_text.translation_values or now_name in get_text.translation._catalog:
            not_system_error.draw()
            continue
        if now_name in cache.npc_name_data:
            if now_name == target_data.name:
                if target_data.nick_name != "":
                    cache.npc_nickname_data.remove(target_data.nick_name)
                target_data.nick_name = ""
                break
            not_name_error.draw()
            continue
        if now_name in cache.npc_nickname_data:
            if now_name == target_data.nick_name:
                break
            not_name_error.draw()
            continue
        if target_data.nick_name != "":
            cache.npc_nickname_data.remove(target_data.nick_name)
        target_data.nick_name = now_name
        cache.npc_nickname_data.add(now_name)
        break


@handle_instruct.add_instruct(
    constant.Instruct.SEE_ACHIEVE,
    constant.InstructType.SYSTEM,
    _("查看成就"),
    {},
)
def handle_see_achieve():
    """处理查看成就指令"""
    py_cmd.clr_cmd()
    now_draw = achieve_panel.AchievePanel(width, True)
    now_draw.draw()


@handle_instruct.add_instruct(constant.Instruct.SAVE, constant.InstructType.SYSTEM, _("读写存档"), {})
def handle_save():
    """处理读写存档指令"""
    py_cmd.clr_cmd()
    now_panel = see_save_info_panel.SeeSaveListPanel(width, 1)
    now_panel.draw()


@handle_instruct.add_instruct(constant.Instruct.OBSERVE_ON, constant.InstructType.SYSTEM, _("开启看海模式"), {})
def handle_observe_on():
    """处理开启看海模式指令"""
    pass


@handle_instruct.add_instruct(constant.Instruct.OBSERVE_OFF, constant.InstructType.SYSTEM, _("关闭看海模式"), {})
def handle_observe_off():
    """处理关闭看海模式指令"""
    pass


@handle_instruct.add_instruct(constant.Instruct.DEBUG_ON, constant.InstructType.SYSTEM, _("开启debug模式"), {constant.Premise.DEBUG_OFF})
def handle_debug_on():
    """处理开启debug模式指令"""
    cache.debug = True


@handle_instruct.add_instruct(constant.Instruct.DEBUG_OFF, constant.InstructType.SYSTEM, _("关闭debug模式"), {constant.Premise.DEBUG_ON})
def handle_debug_off():
    """处理关闭debug模式指令"""
    cache.debug = False
