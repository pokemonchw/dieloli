import random
from Script.Core import (
    cache_contorl,
    py_cmd,
    text_loading,
    era_print,
    game_init,
    constant,
)
from Script.Design import attr_calculation, game_time, nature, character
from Script.Panel import creator_character_panel, see_nature_panel
from Script.Core import game_type


def input_name_func():
    """
    请求玩家输入姓名流程
    输入0:进入请求玩家输入昵称流程(玩家姓名为默认或输入姓名流程结果)
    输入1:进入输入姓名流程
    输入2:返回标题菜单
    """
    game_time.init_time()
    cache_contorl.character_data[0] = game_type.Character()
    flow_return = 0
    while 1:
        flow_return = creator_character_panel.input_name_panel()
        if flow_return == 0:
            py_cmd.clr_cmd()
            break
        elif flow_return == 1:
            py_cmd.clr_cmd()
            creator_character_panel.start_input_name_panel()
            py_cmd.clr_cmd()
        else:
            break
    if flow_return == 0:
        input_nick_name_func()
    else:
        era_print.next_screen_print()
        cache_contorl.now_flow_id = "title_frame"


def input_nick_name_func():
    """
    请求玩家输入昵称流程
    输入0:进入请求玩家输入自称流程(玩家昵称为默认或输入玩家昵称流程结果)
    输入1:进入输入昵称流程
    输入2:使用玩家姓名作为昵称
    输入3:返回请求输入姓名流程
    """
    flow_return = creator_character_panel.input_nick_name_panel()
    if flow_return == 0:
        py_cmd.clr_cmd()
        input_self_name_func()
    elif flow_return == 1:
        py_cmd.clr_cmd()
        creator_character_panel.start_input_nick_name_panel()
        py_cmd.clr_cmd()
        input_nick_name_func()
    elif flow_return == 2:
        py_cmd.clr_cmd()
        cache_contorl.character_data[0].nick_name = cache_contorl.character_data[0].name
        input_nick_name_func()
    elif flow_return == 3:
        py_cmd.clr_cmd()
        input_name_func()


def input_self_name_func():
    """
    请求玩家输入自称流程
    输入0:进入请求玩家输入性别流程(玩家自称为默认，或输入自称流程结果)
    输入1:进入输入自称流程
    输入2:返回请求输入昵称流程
    """
    flow_return = creator_character_panel.input_self_name_panel()
    if flow_return == 0:
        py_cmd.clr_cmd()
        input_sex_confirm_func()
    elif flow_return == 1:
        py_cmd.clr_cmd()
        creator_character_panel.start_input_self_name()
        py_cmd.clr_cmd()
        input_self_name_func()
    elif flow_return == 2:
        era_print.line_feed_print()
        py_cmd.clr_cmd()
        input_nick_name_func()


def input_sex_confirm_func():
    """
    请求玩家输入性别流程
    输入0:进入询问是否进行详细设置流程(玩家性别为默认，或请求选择性别流程结果)
    输入1:进入选择性别流程
    输入2:返回请求输入自称流程
    """
    flow_return = creator_character_panel.input_sex_panel()
    if flow_return == 0:
        py_cmd.clr_cmd()
        attribute_generation_branch_func()
    elif flow_return == 1:
        py_cmd.clr_cmd()
        input_sex_choice_func()
    elif flow_return == 2:
        py_cmd.clr_cmd()
        input_self_name_func()


def input_sex_choice_func():
    """
    玩家选择性别流程
    输入0-3:选择对应性别(Man/Woman/Futa/Asexual)
    输入4:随机选择一个性别
    输入5:返回请求输入性别流程
    """
    sex = list(text_loading.get_text_data(constant.FilePath.ROLE_PATH, "Sex").keys())
    sex_max = len(sex)
    flow_return = creator_character_panel.input_sex_choice_panel()
    if flow_return in range(0, sex_max):
        sex_atr = sex[flow_return]
        cache_contorl.character_data[0].sex = sex_atr
        py_cmd.clr_cmd()
        input_sex_confirm_func()
    elif flow_return == 4:
        rand = random.randint(0, len(sex) - 1)
        sex_atr = sex[rand]
        cache_contorl.character_data[0].sex = sex_atr
        py_cmd.clr_cmd()
        input_sex_confirm_func()
    elif flow_return == 5:
        era_print.list_print()
        py_cmd.clr_cmd()
        input_sex_confirm_func()


def attribute_generation_branch_func():
    """
    询问玩家是否需要详细设置属性流程
    输入0:进入询问玩家年龄段流程
    输入1:进入属性最终确认流程(使用基础模板生成玩家属性)
    输入2:返回请求输入性别流程
    """
    flow_return = creator_character_panel.attribute_generation_branch_panel()
    if flow_return == 0:
        py_cmd.clr_cmd()
        detailed_setting_func_1()
    elif flow_return == 1:
        py_cmd.clr_cmd()
        character.init_attr(0)
        cache_contorl.now_flow_id = "acknowledgment_attribute"
    elif flow_return == 2:
        py_cmd.clr_cmd()
        input_sex_confirm_func()


def detailed_setting_func_1():
    """
    询问玩家年龄模板流程
    """
    flow_retun = creator_character_panel.detailed_setting_1_panel()
    character_age_tem_name = attr_calculation.get_age_tem_list()[flow_retun]
    cache_contorl.character_data[0].age = attr_calculation.get_age(
        character_age_tem_name
    )
    py_cmd.clr_cmd()
    detailed_setting_func_3()


def detailed_setting_func_3():
    """
    询问玩家性经验程度流程
    """
    flow_return = creator_character_panel.detailed_setting_3_panel()
    sex_tem_data_list = list(
        text_loading.get_text_data(
            constant.FilePath.ATTR_TEMPLATE_PATH, "SexExperience"
        ).keys()
    )
    sex_tem_data_list.reverse()
    sex_tem_name = sex_tem_data_list[flow_return]
    cache_contorl.character_data[0].sex_experience_tem = sex_tem_name
    py_cmd.clr_cmd()
    detailed_setting_func_8()


def detailed_setting_func_8():
    """
    询问玩家肥胖程度流程
    """
    flow_return = creator_character_panel.detailed_setting_8_panel()
    weight_tem_data = text_loading.get_text_data(
        constant.FilePath.ATTR_TEMPLATE_PATH, "WeightTem"
    )
    weight_tem_list = list(weight_tem_data.keys())
    weight_tem = weight_tem_list[int(flow_return)]
    cache_contorl.character_data[0].weigt_tem = weight_tem
    cache_contorl.character_data[0].bodyfat_tem = weight_tem
    enter_character_nature_func()


def enter_character_nature_func():
    """
    请求玩家确认性格流程
    """
    character.init_attr(0)
    while 1:
        py_cmd.clr_cmd()
        creator_character_panel.enter_character_nature_head()
        input_s = see_nature_panel.see_character_nature_change_panel(0)
        input_s += creator_character_panel.enter_character_nature_end()
        yrn = game_init.askfor_all(input_s)
        if yrn in cache_contorl.character_data[0].nature:
            if cache_contorl.character_data[0].nature[yrn] < 50:
                cache_contorl.character_data[0].nature[yrn] = random.uniform(50, 100)
            else:
                cache_contorl.character_data[0].nature[yrn] = random.uniform(0, 50)
        elif int(yrn) == 0:
            character.init_attr(0)
            cache_contorl.now_flow_id = "acknowledgment_attribute"
            break
        elif int(yrn) == 1:
            cache_contorl.character_data[0].nature = nature.get_random_nature()
