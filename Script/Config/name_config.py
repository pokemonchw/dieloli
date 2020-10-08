import os
from typing import Dict
from Script.Core import json_handle


name_json_path = os.path.join("data","NameIndex.json")
""" 原始名字数据文件路径 """
family_json_path = os.path.join("data","FamilyIndex.json")
""" 原始姓氏数据文件路径 """
man_name_data:Dict[str,int] = {}
"""
男性名字权重数据
名字:权重
"""
woman_name_data:Dict[str,int] = {}
"""
女性名字权重数据
名字:权重
"""
family_data:Dict[str,int] = {}
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
    man_name_data = name_data["Boys"]
    woman_name_data = name_data["Girls"]
    family_data = json_handle.load_json(family_json_path)["FamilyNameList"]
