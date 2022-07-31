import os
import random
from functools import wraps
from typing import List
from types import FunctionType
from Script.Core import get_text, constant, game_type, cache_control, flow_handle
from Script.Design import (
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
)
from Script.UI.Moudle import panel, draw
from Script.UI.Panel import see_character_info_panel
from Script.Config import normal_config, game_config
from Script.Core import json_handle


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


creator_character_flow_json_path = os.path.join(
    "data/json/UI/Flow", "CreatorCharacterFlow.json")
""" 角色创造者数据文件路径 """


@handle_panel.add_panel(constant.Panel.CREATOR_CHARACTER)
def creator_character_panel():
    """创建角色面板"""
    game_time.init_time()
    cache.character_data[0] = game_type.Character()
    character_handle.init_character_list()
    while 1:
        if input_name_panel():
            character.init_attr(0)
            game_start()
            if confirm_character_attr_panel():
                break
        cache.character_data[0] = game_type.Character()
    cache.now_panel_id = constant.Panel.IN_SCENE


def game_start():
    """初始化游戏数据"""
    character_handle.init_character_dormitory()
    character_handle.init_character_position()
    course.init_phase_course_hour()
    interest.init_character_interest()
    course.init_character_knowledge()
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


def confirm_character_attr_panel():
    """确认角色属性面板"""
    confirm_character = json_handle.load_json(creator_character_flow_json_path)[
        confirm_character_attr_panel.__name__]

    now_attr_panel = see_character_info_panel.SeeCharacterInfoPanel(0, width)
    askfor_panel = panel.OneMessageAndSingleColumnButton()
    while 1:
        line_feed_draw.draw()
        now_attr_panel.draw()
        ask_list = []
        ask_list.extend(now_attr_panel.return_list)
        now_line = draw.LineDraw("~", width)
        now_line.draw()
        askfor_list = confirm_character["ask_list"]
        start_id = 0
        now_id_judge = 0
        now_id_list = []
        for now_id in ask_list:
            if now_id.isdigit():
                now_id_judge = 1
                now_id_list.append(int(now_id))
        if now_id_judge:
            start_id = max(now_id_list) + 1
        askfor_panel.set(askfor_list, confirm_character["ask"], start_id)
        askfor_panel.draw()
        askfor_panel_return_list = askfor_panel.get_return_list()
        ask_list.extend(askfor_panel_return_list.keys())
        yrn = flow_handle.askfor_all(ask_list)
        if yrn in askfor_panel_return_list:
            return askfor_panel_return_list[yrn] == askfor_list[0]


def input_name_panel() -> bool:
    """
    输入角色名面板
    Return arguments:
    bool -- 完成角色创建校验
    """
    input_name = json_handle.load_json(creator_character_flow_json_path)[
        input_name_panel.__name__]

    character_data = cache.character_data[0]
    ask_name_panel = panel.AskForOneMessage()
    ask_name_panel.set(input_name["ask"], 10)
    line_feed_draw.draw()
    line.draw()
    not_num_error = draw.NormalDraw()
    not_num_error.text = input_name["num_error"]
    not_system_error = draw.NormalDraw()
    not_system_error.text = input_name["system_error"]
    not_name_error = draw.NormalDraw()
    not_name_error.text = input_name["name_error"]
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
        create_judge = input_nick_name_panel()
        break
    return create_judge


def input_nick_name_panel() -> bool:
    """
    输入角色昵称面板
    Return arguments:
    bool -- 完成角色创建校验
    """
    input_nick_name = json_handle.load_json(creator_character_flow_json_path)[
        input_nick_name_panel.__name__]

    create_judge = 0
    character_data = cache.character_data[0]
    ask_nick_name_panel = panel.AskForOneMessage()
    ask_nick_name_panel.set(
        input_nick_name["ask"].format(
            character_name=character_data.name), 10
    )
    line_feed_draw.draw()
    line.draw()
    not_num_error = draw.NormalDraw()
    not_num_error.text = input_nick_name["num_error"]
    not_system_error = draw.NormalDraw()
    not_system_error.text = input_nick_name["system_error"]
    while 1:
        nick_name = ask_nick_name_panel.draw()
        if nick_name.isdigit():
            not_num_error.draw()
            continue
        if nick_name in get_text.translation_values or nick_name in get_text.translation._catalog:
            not_system_error.draw()
            continue
        character_data.nick_name = nick_name
        create_judge = input_sex_panel()
        break
    return create_judge


def input_sex_panel() -> bool:
    """
    选择性别面板
    Return arguments:
    bool -- 完成角色创建校验
    """
    input_sex = json_handle.load_json(creator_character_flow_json_path)[
        input_sex_panel.__name__]

    character_data: game_type.Character = cache.character_data[0]
    sex_list = [
        game_config.config_sex_tem[x].name for x in game_config.config_sex_tem] + [input_sex["random"]]
    button_panel = panel.OneMessageAndSingleColumnButton()
    button_panel.set(
        sex_list,
        input_sex["ask"].format(
            character_nick_name=character_data.nick_name),
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
    if character_data.sex in {1, 2}:
        character_data.chest_tem = attr_calculation.get_rand_npc_chest_tem()
    create_judge = input_setting_panel()
    return create_judge


def input_setting_panel() -> bool:
    """
    询问设置详细信息面板
    Return arguments:
    bool -- 完成角色创建流程
    """
    input_setting = json_handle.load_json(creator_character_flow_json_path)[
        input_setting_panel.__name__]

    character_data = cache.character_data[0]
    ask_list = input_setting["ask_list"]
    button_panel = panel.OneMessageAndSingleColumnButton()
    button_panel.set(
        ask_list,
        input_setting["ask"].format(
            character_nick_name=character_data.nick_name
        ),
    )
    return_list = button_panel.get_return_list()
    line_feed_draw.draw()
    line.draw()
    button_panel.draw()
    ans = flow_handle.askfor_all(return_list.keys())
    if int(ans):
        return 1
    return input_setting_now()


def input_setting_now() -> bool:
    """启动详细信息设置"""
    panel_list = random.sample(setting_panel_data, len(setting_panel_data))
    for now_panel in panel_list:
        line_feed_draw.draw()
        line.draw()
        now_panel()
    return 1


setting_panel_data: List[FunctionType] = []
""" 设置详细信息面板数据 """


def add_setting_panel() -> FunctionType:
    """
    添加创建角色时设置详细信息面板
    Return arguments:
    FunctionType -- 面板对象处理函数
    """

    def decorator(func):
        @wraps(func)
        def return_wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        setting_panel_data.append(return_wrapper)
        return return_wrapper

    return decorator


@add_setting_panel()
def setting_age_tem_panel():
    """设置年龄模板"""
    setting_age_tem = json_handle.load_json(creator_character_flow_json_path)[
        setting_age_tem_panel.__name__]

    character_data: game_type.Character = cache.character_data[0]
    message = setting_age_tem["aske"].format(
        character_nick_name=character_data.nick_name
    )
    ask_list = setting_age_tem["ask_list"]
    button_panel = panel.OneMessageAndSingleColumnButton()
    button_panel.set(ask_list, message)
    return_list = button_panel.get_return_list()
    button_panel.draw()
    ans = flow_handle.askfor_all(return_list.keys())
    character_data.age = attr_calculation.get_age(int(ans))


@add_setting_panel()
def setting_weight_panel():
    """设置体重模板"""
    setting_weight = json_handle.load_json(creator_character_flow_json_path)[
        setting_weight_panel.__name__]

    character_data = cache.character_data[0]
    message = setting_weight["ask"].format(
        character_nick_name=character_data.nick_name
    )
    ask_list = setting_weight["ask_list"]
    button_panel = panel.OneMessageAndSingleColumnButton()
    button_panel.set(ask_list, message)
    return_list = button_panel.get_return_list()
    button_panel.draw()
    ans = flow_handle.askfor_all(return_list.keys())
    character_data.weight_tem = int(ans)


@add_setting_panel()
def setting_sex_experience_panel():
    """设置性经验模板"""
    setting_sex_experience = json_handle.load_json(creator_character_flow_json_path)[
        setting_sex_experience_panel.__name__]

    character_data = cache.character_data[0]
    message = setting_sex_experience["ask"].format(
        character_nick_name=character_data.nick_name
    )
    ask_list = setting_sex_experience["ask_list"]
    button_panel = panel.OneMessageAndSingleColumnButton()
    button_panel.set(ask_list, message)
    return_list = button_panel.get_return_list()
    button_panel.draw()
    ans = flow_handle.askfor_all(return_list.keys())
    character_data.sex_experience_tem = int(ans)


@add_setting_panel()
def setting_nature_0_panel():
    """设置性格倾向:活跃"""
    setting_nature_0 = json_handle.load_json(creator_character_flow_json_path)[
        setting_nature_0_panel.__name__]

    character_data = cache.character_data[0]
    message = setting_nature_0["ask"].format(
        character_nick_name=character_data.nick_name
    )
    ask_list = setting_nature_0["ask_list"]
    button_panel = panel.OneMessageAndSingleColumnButton()
    button_panel.set(ask_list, message)
    return_list = button_panel.get_return_list()
    button_panel.draw()
    ans = flow_handle.askfor_all(return_list.keys())
    character_data.nature[0] = abs(random.randint(0, 100) - int(ans) * 50)


@add_setting_panel()
def setting_nature_1_panel():
    """设置性格倾向:合群"""
    setting_nature_1 = json_handle.load_json(creator_character_flow_json_path)[
        setting_nature_1_panel.__name__]

    character_data = cache.character_data[0]
    message = setting_nature_1["ask"].format(
        character_nick_name=character_data.nick_name
    )
    ask_list = setting_nature_1["ask_list"]
    button_panel = panel.OneMessageAndSingleColumnButton()
    button_panel.set(ask_list, message)
    return_list = button_panel.get_return_list()
    button_panel.draw()
    ans = flow_handle.askfor_all(return_list.keys())
    character_data.nature[1] = abs(random.randint(0, 100) - int(ans) * 50)


@add_setting_panel()
def setting_nature_2_panel():
    """设置性格倾向:乐观"""
    setting_nature_2 = json_handle.load_json(creator_character_flow_json_path)[
        setting_nature_2_panel.__name__]

    character_data = cache.character_data[0]
    message = setting_nature_2["ask"].format(
        character_nick_name=character_data.nick_name
    )
    ask_list = setting_nature_2["ask_list"]
    button_panel = panel.OneMessageAndSingleColumnButton()
    button_panel.set(ask_list, message)
    return_list = button_panel.get_return_list()
    button_panel.draw()
    ans = flow_handle.askfor_all(return_list.keys())
    character_data.nature[2] = abs(random.randint(0, 100) - int(ans) * 50)


@add_setting_panel()
def setting_nature_3_panel():
    """设置性格倾向:守信"""
    setting_nature_3 = json_handle.load_json(creator_character_flow_json_path)[
        setting_nature_3_panel.__name__]

    character_data = cache.character_data[0]
    message = setting_nature_3["ask"]
    ask_list = setting_nature_3["ask_list"]
    button_panel = panel.OneMessageAndSingleColumnButton()
    button_panel.set(ask_list, message)
    return_list = button_panel.get_return_list()
    button_panel.draw()
    ans = flow_handle.askfor_all(return_list.keys())
    character_data.nature[3] = abs(random.randint(0, 100) - int(ans) * 50)


@add_setting_panel()
def setting_nature_4_panel():
    """设置性格区间:无私"""
    setting_nature_4 = json_handle.load_json(creator_character_flow_json_path)[
        setting_nature_4_panel.__name__]

    character_data = cache.character_data[0]
    message = setting_nature_4["ask"]
    ask_list = setting_nature_4["ask_list"]
    button_panel = panel.OneMessageAndSingleColumnButton()
    button_panel.set(ask_list, message)
    return_list = button_panel.get_return_list()
    button_panel.draw()
    ans = flow_handle.askfor_all(return_list.keys())
    character_data.nature[4] = abs(random.randint(0, 100) - int(ans) * 50)


@add_setting_panel()
def setting_nature_5_panel():
    """设置性格区间:重情"""
    setting_nature_5 = json_handle.load_json(creator_character_flow_json_path)[
        setting_nature_5_panel.__name__]

    character_data = cache.character_data[0]
    message = setting_nature_5["ask"]
    ask_list = setting_nature_5["ask_list"]
    button_panel = panel.OneMessageAndSingleColumnButton()
    button_panel.set(ask_list, message)
    return_list = button_panel.get_return_list()
    button_panel.draw()
    ans = flow_handle.askfor_all(return_list.keys())
    character_data.nature[5] = abs(random.randint(0, 100) - int(ans) * 50)


@add_setting_panel()
def setting_nature_6_panel():
    """设置性格区间:严谨"""
    setting_nature_6 = json_handle.load_json(creator_character_flow_json_path)[
        setting_nature_6_panel.__name__]

    character_data = cache.character_data[0]
    message = setting_nature_6["ask"]
    ask_list = setting_nature_6["ask_list"]
    button_panel = panel.OneMessageAndSingleColumnButton()
    button_panel.set(ask_list, message)
    return_list = button_panel.get_return_list()
    button_panel.draw()
    ans = flow_handle.askfor_all(return_list.keys())
    character_data.nature[6] = abs(random.randint(0, 100) - int(ans) * 50)


@add_setting_panel()
def setting_nature_7_panel():
    """设置性格区间:自律"""
    setting_nature_7 = json_handle.load_json(creator_character_flow_json_path)[
        setting_nature_7_panel.__name__]

    character_data = cache.character_data[0]
    message = setting_nature_7["ask"].format(
        character_nick_name=character_data.nick_name
    )
    ask_list = setting_nature_7["ask_list"]
    button_panel = panel.OneMessageAndSingleColumnButton()
    button_panel.set(ask_list, message)
    return_list = button_panel.get_return_list()
    button_panel.draw()
    ans = flow_handle.askfor_all(return_list.keys())
    character_data.nature[7] = abs(random.randint(0, 100) - int(ans) * 50)


@add_setting_panel()
def setting_nature_8_panel():
    """设置性格区间:沉稳"""
    setting_nature_8 = json_handle.load_json(creator_character_flow_json_path)[
        setting_nature_8_panel.__name__]

    character_data = cache.character_data[0]
    message = setting_nature_8["ask"]
    ask_list = setting_nature_8["ask_list"]
    button_panel = panel.OneMessageAndSingleColumnButton()
    button_panel.set(ask_list, message)
    return_list = button_panel.get_return_list()
    button_panel.draw()
    ans = flow_handle.askfor_all(return_list.keys())
    character_data.nature[8] = abs(random.randint(0, 100) - int(ans) * 50)


@add_setting_panel()
def setting_nature_9_panel():
    """设置性格区间:决断"""
    setting_nature_9 = json_handle.load_json(creator_character_flow_json_path)[
        setting_nature_9_panel.__name__]

    character_data = cache.character_data[0]
    message = setting_nature_9["ask"].format(
        character_nick_name=character_data.nick_name
    )
    ask_list = setting_nature_9["ask_list"]
    button_panel = panel.OneMessageAndSingleColumnButton()
    button_panel.set(ask_list, message)
    return_list = button_panel.get_return_list()
    button_panel.draw()
    ans = flow_handle.askfor_all(return_list.keys())
    character_data.nature[9] = abs(random.randint(0, 100) - int(ans) * 50)


@add_setting_panel()
def setting_nature_10_panel():
    """设置性格区间:坚韧"""
    setting_nature_10 = json_handle.load_json(creator_character_flow_json_path)[
        setting_nature_10_panel.__name__]

    character_data = cache.character_data[0]
    message = setting_nature_10["ask"]
    ask_list = setting_nature_10["ask_list"]
    button_panel = panel.OneMessageAndSingleColumnButton()
    button_panel.set(ask_list, message)
    return_list = button_panel.get_return_list()
    button_panel.draw()
    ans = flow_handle.askfor_all(return_list.keys())
    character_data.nature[10] = abs(random.randint(0, 100) - int(ans) * 50)


@add_setting_panel()
def setting_nature_11_panel():
    """设置性格区间:机敏"""
    setting_nature_11 = json_handle.load_json(creator_character_flow_json_path)[
        setting_nature_11_panel.__name__]

    character_data = cache.character_data[0]
    message = setting_nature_11["ask"].format(
        character_nick_name=character_data.nick_name
    )
    ask_list = setting_nature_11["ask_list"]
    button_panel = panel.OneMessageAndSingleColumnButton()
    button_panel.set(ask_list, message)
    return_list = button_panel.get_return_list()
    button_panel.draw()
    ans = flow_handle.askfor_all(return_list.keys())
    character_data.nature[11] = abs(random.randint(0, 100) - int(ans) * 50)


@add_setting_panel()
def setting_nature_12_panel():
    """设置性格区间:耐性"""
    setting_nature_12 = json_handle.load_json(creator_character_flow_json_path)[
        setting_nature_12_panel.__name__]

    character_data = cache.character_data[0]
    message = setting_nature_12["ask"]
    ask_list = setting_nature_12["ask_list"]
    button_panel = panel.OneMessageAndSingleColumnButton()
    button_panel.set(ask_list, message)
    return_list = button_panel.get_return_list()
    button_panel.draw()
    ans = flow_handle.askfor_all(return_list.keys())
    character_data.nature[12] = abs(random.randint(0, 100) - int(ans) * 50)


@add_setting_panel()
def setting_nature_13_panel():
    """设置性格区间:爽直"""
    setting_nature_13 = json_handle.load_json(creator_character_flow_json_path)[
        setting_nature_13_panel.__name__]

    character_data = cache.character_data[0]
    message = setting_nature_13["ask"].format(
        character_nick_name=character_data.nick_name
    )
    ask_list = setting_nature_13["ask_list"]
    button_panel = panel.OneMessageAndSingleColumnButton()
    button_panel.set(ask_list, message)
    return_list = button_panel.get_return_list()
    button_panel.draw()
    ans = flow_handle.askfor_all(return_list.keys())
    character_data.nature[13] = abs(random.randint(0, 100) - int(ans) * 50)
