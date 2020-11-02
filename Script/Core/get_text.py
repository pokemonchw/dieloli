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
    return gettext.translation(
        "dieloli", po_data, [normal_config.config_normal.language, "zh_CN"]
    ).gettext


_: FunctionType = getLocStrings()
""" 翻译api """
