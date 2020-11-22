# -*- coding: UTF-8 -*-
import gettext
import os
from types import FunctionType
from Script.Config import normal_config


def getLocStrings() -> FunctionType:
    """
    翻译api初始化
    Return arguments:
    FunctionType -- 翻译处理函数对象
    """
    po_data = os.path.join("data", "po")
    try:
        return gettext.translation("dieloli", po_data, [normal_config.config_normal.language, "zh_CN"]).gettext
    except FileNotFoundError:
        raise Exception(
            "请先将{}中的 *.po 文件转换为 *.mo 文件！（推荐使用 Poedit 工具）".format(po_data))


_: FunctionType = getLocStrings()
""" 翻译api """
