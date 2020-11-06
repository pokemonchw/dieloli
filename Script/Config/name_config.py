import os
from typing import Dict
from Script.Core import json_handle,cache_contorl,value_handle


name_json_path = os.path.join("data", "NameIndex.json")
""" 原始名字数据文件路径 """
family_json_path = os.path.join("data", "FamilyIndex.json")
""" 原始姓氏数据文件路径 """
man_name_data: Dict[str, int] = {}
"""
男性名字权重数据
名字:权重
"""
woman_name_data: Dict[str, int] = {}
"""
女性名字权重数据
名字:权重
"""
family_data: Dict[str, int] = {}
"""
姓氏权重配置数据
姓氏:权重
"""


def init_name_data():
    """ 载入json内姓名配置数据 """
    global man_name_data
    global woman_name_data
    global family_data
    name_data = json_handle.load_json(name_json_path)
    init_name_region(name_data["Boys"],0)
    init_name_region(name_data["Girls"],1)
    family_data = json_handle.load_json(family_json_path)["FamilyNameList"]
    init_name_region(family_data,2)

def init_name_region(name_data: Dict[str, int], man_judge: int):
    """
    初始化性别名字随机权重
    Keyword arguments:
    name_data -- 名字数据
    man_judge -- 类型校验(0:男,1:女,2:姓)
    """
    region_list = value_handle.get_region_list(name_data)
    if man_judge == 0:
        cache_contorl.boys_region_list = region_list
        cache_contorl.boys_region_int_list = list(map(int, region_list))
    elif man_judge == 1:
        cache_contorl.girls_region_list = region_list
        cache_contorl.girls_region_int_list = list(map(int, region_list))
    else:
        cache_contorl.family_region_list = region_list
        cache_contorl.family_region_int_list = list(map(int, region_list))
