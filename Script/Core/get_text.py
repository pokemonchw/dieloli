# -*- coding: UTF-8 -*-
import gettext
import os
from types import FunctionType


_:FunctionType = gettext.gettext
""" 翻译api """

def init_gettext():
    """
    初始化gettext设置
    Keyword arguments:
    language -- 语言
    """
    po_data = os.path.join("data","po")
    gettext.bindtextdomain("dieloli", po_data)
    gettext.textdomain("dieloli")
