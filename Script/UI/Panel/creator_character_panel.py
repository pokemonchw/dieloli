import random
from types import FunctionType
from Script.Core import get_text,constant,game_type,cache_contorl,flow_handle
from Script.Design import handle_panel
from Script.UI.Moudle import panel,draw
from Script.Config import normal_config,game_config

_:FunctionType = get_text._
""" 翻译api """
width = normal_config.config_normal.text_width
""" 屏幕宽度 """
line_feed_draw = draw.NormalDraw()
""" 绘制换行对象 """
line_feed_draw.text = "\n"
line = draw.LineDraw("=",width)
""" 标题线绘制对象 """

@handle_panel.add_panel(constant.Panel.CREATOR_CHARACTER)
def creator_character_panel():
    """ 创建角色面板 """
    cache_contorl.character_data[0] = game_type.Character()
    while 1:
        if input_name_panel():
            break

def input_name_panel() -> bool:
    """
    输入角色名面板
    Return arguments:
    bool -- 完成角色创建校验
    """
    character_data = cache_contorl.character_data[0]
    ask_name_panel = panel.AskForOneMessage()
    ask_name_panel.set(_("请问能告诉我你的名字吗？"),10)
    line_feed_draw.draw()
    line.draw()
    not_num_error = draw.NormalDraw()
    not_num_error.text = _("角色名不能为纯数字，请重新输入")
    create_judge = 0
    while 1:
        now_name = ask_name_panel.draw()
        if now_name.isalnum():
            not_num_error.draw()
            continue
        character_data.name = now_name
        create_judge = input_nick_name_panel()
        break
    return create_judge

def input_nick_name_panel() -> bool:
    """
    输入角色昵称面板
    Return arguments:
    bool -- 完成角色创建校验
    """
    create_judge = 0
    character_data = cache_contorl.character_data[0]
    ask_nick_name_panel = panel.AskForOneMessage()
    ask_nick_name_panel.set(_(f"该怎么称呼{character_data.name}好呢？"),10)
    line_feed_draw.draw()
    line.draw()
    not_num_error = draw.NormalDraw()
    not_num_error.text = _("角色昵称不能为纯数字，请重新输入")
    while 1:
        nick_name = ask_nick_name_panel.draw()
        if int(nick_name):
            not_num_error.draw()
            continue
        character_data.nick_name = nick_name
        create_judge = input_sex_panel()
    return create_judge

def input_sex_panel() -> bool:
    """
    选择性别面板
    Return arguments:
    bool -- 完成角色创建校验
    """
    create_judge = 0
    character_data = cache_contorl.character_data[0]
    sex_list = [game_config.config_sex_tem[x].name for x in game_config.config_sex_tem] + [_("随机")]
    button_panel = panel.OneMessageAndSingleColumnButton()
    button_panel.set(sex_list,_(f"那么{character_data.nick_name}的性别是？"),width)
    return_list = button_panel.get_return_list()
    button_panel.draw()
    ans = flow_handle.askfor_all(return_list.keys())
    now_id = int(ans)
    if now_id == len(return_list) - 1:
        now_id = random.randint(0,now_id-1)
    character_data.sex = now_id
    create_judge = input_setting_panel()
    return create_judge


def input_setting_panel() -> bool:
    """
    询问设置详细信息面板
    Return arguments:
    bool -- 完成角色创建流程
    """
    create_judge = 0
    character_data = cache_contorl.character_data[0]
    ask_list = [_("是"),_("否")]
    button_panel = panel.OneMessageAndSingleColumnButton()
    button_panel.set(ask_list,_(f"是否需要设置详细属性呢？将会随机抽取十道题目供{character_data.nick_name}进行选择。"),width)
    return_list = button_panel.get_return_list()
    button_panel.draw()
    ans = flow_handle.askfor_all(return_list.keys())
    if int(ans):
        return ans
    return 0
