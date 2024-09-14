import random
from functools import wraps
from typing import List
from types import FunctionType
from Script.Core import get_text, game_type, cache_control, flow_handle, py_cmd
from Script.Design import (
    constant,
    handle_panel,
    character,
    character_handle,
    course,
    cooking,
    map_handle,
    interest,
    attr_calculation,
    game_time,
    clothing,
    club_handle,
    handle_achieve,
    weather,
    handle_adv,
)
from Script.UI.Moudle import panel, draw
from Script.UI.Panel import see_character_info_panel, change_nature_panel
from Script.Config import normal_config, game_config, map_config

cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """
_: FunctionType = get_text._
""" 翻译api """
width = normal_config.config_normal.text_width
""" 屏幕宽度 """
line_feed_draw = draw.NormalDraw()
""" 绘制换行对象 """
line_feed_draw.text = "\n"
line = draw.LineDraw("=", width)
""" 标题线绘制对象 """


@handle_panel.add_panel(constant.Panel.CREATOR_CHARACTER)
def creator_character_panel():
    """创建角色面板"""
    cache.__init__()
    map_config.init_map_data()
    game_time.init_time()
    cache.character_data[0] = game_type.Character()
    character_handle.init_character_list()
    character.init_attr(0)
    game_start()
    confirm_character_attr_panel()
    character_handle.init_character_position()
    cache_control.achieve.create_npc_index += len(cache.character_data)
    handle_achieve.check_all_achieve()
    weather.handle_weather(False)
    cache.now_panel_id = constant.Panel.IN_SCENE


def game_start():
    """初始化游戏数据"""
    character_handle.init_no_character_scene()
    character_handle.init_character_dormitory()
    course.init_phase_course_hour()
    interest.init_character_interest()
    course.init_all_character_knowledge()
    course.init_class_teacher()
    course.init_class_time_table()
    course.init_teacher_table()
    cooking.init_recipes()
    cooking.init_restaurant_data()
    clothing.init_clothing_shop_data()
    character_position = cache.character_data[0].position
    map_handle.character_move_scene(["0"], character_position, 0)
    cache.school_longitude = random.uniform(120.9, 122.12)
    cache.school_latitude = random.uniform(30.7, 31.53)
    club_handle.init_club_data()
    for i in cache.character_data:
        attr_calculation.init_character_hp_and_mp(i)
    handle_adv.handle_all_adv_npc()


def confirm_character_attr_panel():
    """确认角色属性面板"""
    attr_panel_id = ""
    while 1:
        py_cmd.clr_cmd()
        now_attr_panel = see_character_info_panel.SeeCharacterInfoPanel(0, width)
        if attr_panel_id != "":
            now_attr_panel.change_panel(attr_panel_id)
        askfor_panel = panel.OneMessageAndSingleColumnButton()
        line_feed_draw.draw()
        now_attr_panel.draw()
        ask_list = []
        ask_list.extend(now_attr_panel.return_list)
        now_line = draw.LineDraw("~", width)
        now_line.draw()
        change_attr_button_list = [_("[更改姓名]"), _("[更改性别]"), _("[修正年龄]"), _("[修正身材]"), _("[修改性格]")]
        ask_change_attr_draw = panel.CenterDrawButtonListPanel()
        ask_change_attr_draw.set(change_attr_button_list, change_attr_button_list, width, 5)
        ask_change_attr_draw.draw()
        ask_list.extend(change_attr_button_list)
        now_line.draw()
        askfor_list = [_("就这样开始新的人生吧")]
        start_id = 0
        now_id_judge = 0
        now_id_list = []
        for now_id in ask_list:
            if now_id.isdigit():
                now_id_judge = 1
                now_id_list.append(int(now_id))
        if now_id_judge:
            start_id = max(now_id_list) + 1
        askfor_panel.set(askfor_list, _("就这样了可以吗?"), start_id)
        askfor_panel.draw()
        askfor_panel_return_list = askfor_panel.get_return_list()
        ask_list.extend(askfor_panel_return_list.keys())
        yrn = flow_handle.askfor_all(ask_list)
        if yrn == "0":
            break
        elif yrn == change_attr_button_list[0]:
            change_name()
        elif yrn == change_attr_button_list[1]:
            input_sex_panel()
        elif yrn == change_attr_button_list[2]:
            setting_age_tem_panel()
        elif yrn == change_attr_button_list[3]:
            setting_weight_panel()
        elif yrn == change_attr_button_list[4]:
            change_nature_panel.ChangeNaturePanel(width).draw()
        attr_panel_id = now_attr_panel.now_panel


def change_name():
    """询问更改名字"""
    py_cmd.clr_cmd()
    character_data = cache.character_data[0]
    ask_name_panel = panel.AskForOneMessage()
    ask_name_panel.set(_("请问能告诉我你的名字吗？"), 10)
    line_feed_draw.draw()
    line.draw()
    not_num_error = draw.NormalDraw()
    not_num_error.text = _("角色名不能为纯数字，请重新输入\n")
    not_system_error = draw.NormalDraw()
    not_system_error.text = _("角色名不能为系统保留字，请重新输入\n")
    not_name_error = draw.NormalDraw()
    not_name_error.text = _("已有角色使用该姓名，请重新输入\n")
    create_judge = 0
    while 1:
        now_name = ask_name_panel.draw()
        if now_name.isdigit():
            not_num_error.draw()
            continue
        if now_name in get_text.translation_values or now_name in get_text.translation._catalog:
            not_system_error.draw()
            continue
        if now_name in cache.npc_name_data:
            not_name_error.draw()
            continue
        character_data.name = now_name
        break


def input_sex_panel() -> bool:
    """
    选择性别面板
    Return arguments:
    bool -- 完成角色创建校验
    """
    py_cmd.clr_cmd()
    character_data: game_type.Character = cache.character_data[0]
    sex_list = [game_config.config_sex_tem[x].name for x in game_config.config_sex_tem] + [_("随机")]
    button_panel = panel.OneMessageAndSingleColumnButton()
    button_panel.set(
        sex_list,
        _("那么{character_nick_name}的性别是？").format(character_nick_name=character_data.nick_name),
    )
    return_list = button_panel.get_return_list()
    line_feed_draw.draw()
    line.draw()
    button_panel.draw()
    ans = flow_handle.askfor_all(return_list.keys())
    now_id = int(ans)
    if now_id == len(return_list) - 1:
        now_id = random.randint(0, now_id - 1)
    character_data.sex = now_id
    character_handle.init_character_dormitory()
    character.init_character_height(0)
    character.init_character_weight_and_bodyfat(0)
    character.init_character_measurements(0)


def setting_age_tem_panel():
    """设置年龄模板"""
    py_cmd.clr_cmd()
    character_data: game_type.Character = cache.character_data[0]
    message = _("{character_nick_name}是一个小孩子吗？").format(
        character_nick_name=character_data.nick_name
    )
    ask_list = [
        _("嘎呜～嘎呜～"),
        _("才，才不是小孩子！"),
        _("已经成年了哦～"),
        _("我也想回到小时候呢～"),
        _("你说什么？我听不清～"),
    ]
    line_feed_draw.draw()
    line.draw()
    button_panel = panel.OneMessageAndSingleColumnButton()
    button_panel.set(ask_list, message)
    return_list = button_panel.get_return_list()
    button_panel.draw()
    ans = flow_handle.askfor_all(return_list.keys())
    character_data.age = attr_calculation.get_age(int(ans))
    character_handle.init_character_dormitory()
    character.init_character_birthday(0)
    character.init_character_end_age(0)
    character.init_character_height(0)
    character.init_character_weight_and_bodyfat(0)
    character.init_character_measurements(0)
    course.init_character_knowledge(0)
    attr_calculation.init_character_hp_and_mp(0)
    course.init_class_teacher()


def setting_weight_panel():
    """设置体重模板"""
    py_cmd.clr_cmd()
    character_data = cache.character_data[0]
    message = _("{character_nick_name}对自己的体重有自信吗？").format(
        character_nick_name=character_data.nick_name
    )
    ask_list = [
        _("很轻，就像一张纸一样，风一吹就能飘起来。"),
        _("普普通通，健康的身材。"),
        _("略沉，不过从外表不怎么能够看得出来。"),
        _("肉眼可辨的比别人要胖很多。"),
        _("人类的极限，看上去像是相扑选手一样。"),
    ]
    line_feed_draw.draw()
    line.draw()
    button_panel = panel.OneMessageAndSingleColumnButton()
    button_panel.set(ask_list, message)
    return_list = button_panel.get_return_list()
    button_panel.draw()
    ans = flow_handle.askfor_all(return_list.keys())
    character_data.weight_tem = int(ans)
    character_data.bodyfat_tem = int(ans)
    character.init_character_weight_and_bodyfat(0)
    character.init_character_measurements(0)
    attr_calculation.init_character_hp_and_mp(0)

