# -*- coding: UTF-8 -*-
import gettext
import os
from types import FunctionType
from Script.Config import normal_config

po_data = os.path.join("data", "po")
""" po文件路径 """

def _translation(message: str) -> str:
    """ 容错翻译函数 """
    return message

try:
    if "language" in normal_config.config_normal.__dict__:
        translation: gettext.GNUTranslations = gettext.translation(
            "dieloli", po_data, [normal_config.config_normal.language, "zh_CN"]
        )
        """ 翻译对象类型 """
    else:
        translation: gettext.GNUTranslations = gettext.translation(
            "dieloli",  po_data, ["zh_CN", "zh_CN"]
        )
        """ 翻译对象类型 """
    translation_values = set(translation._catalog.values())
    """ 翻译后的文本数据 """
    _: FunctionType = translation.gettext
    """ 翻译api """
except:
    translation_values = set()
    """ 翻译后的文本数据 """
    _: FunctionType = _translation
    """ 翻译api """
