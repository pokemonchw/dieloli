# -*- coding: UTF-8 -*-
import gettext
import os
import polib
from types import FunctionType
from Script.Config import normal_config

po_data = os.path.join("data", "po")
""" po文件路径 """
_: FunctionType = None
""" 翻译api """
translation_values = set()
""" 翻译后文本数据 """
translation: gettext.GNUTranslations = None
""" 原始翻译类 """


def init_translation():
    """ 初始化翻译api """
    global _
    global translation_values
    global translation
    _.__init__()
    if "language" in normal_config.config_normal.__dict__:
        translation = gettext.translation(
            "dieloli", po_data, [normal_config.config_normal.language, "zh_CN"]
        )
        """ 翻译对象类型 """
    else:
        translation = gettext.translation(
            "dieloli",  po_data, ["zh_CN", "zh_CN"]
        )
        """ 翻译对象类型 """
    translation.install()
    translation_values = set(translation._catalog.values())
    _ = translation.gettext


def rebuild_mo():
    """ 重新当前语言的构造mo文件 """
    now_language = "zh_CN"
    if "language" in normal_config.config_normal.__dict__:
        now_language = normal_config.config_normal.language
    now_dir = os.path.join(po_data,now_language, "LC_MESSAGES")
    po_file = os.path.join(now_dir, "dieloli.po")
    mo_file = os.path.join(now_dir, "dieloli.mo")
    po = polib.pofile(po_file)
    po.save_as_mofile(mo_file)
