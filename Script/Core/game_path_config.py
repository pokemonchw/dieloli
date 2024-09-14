import os
import sys
import platform
import appdirs


def resource_path(relative_path: str) -> str:
    """
    获取资源文件的绝对路径，兼容 PyInstaller 打包后的 exe 和未打包的情况
    Keyword arguments:
    relative_path -- 原始路径
    Return arguments:
    str -- 处理后的路径
    """
    if hasattr(sys, '_MEIPASS'):
        # 被打包成 exe 后
        base_path = sys._MEIPASS
    else:
        # 未打包，正常运行
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


config_info = "[game]\ngame_name = 死亡萝莉\nauthor = 任悠云\nverson_time = @2017-2024\nbackground = #000\nlanguage = zh_CN\nwindow_width = 1108\nwindow_hight = 1000\ntextbox_width = 120\ntextbox_hight = 60\ntext_width = 120\ntext_hight = 50\ninputbox_width = 133\nmax_save = 100\nsave_page = 10\ncharacterlist_show = 10\ntext_wait = 0\nhome_url = https://github.com/pokemonchw/dieloli\nlicenses_url = http://creativecommons.org/licenses/by-nc-sa/2.0/\nproportion_teacher = 1\nproportion_student = 23\nthreading_pool_max = 20\ninsceneseeplayer_max = 20\nseecharacterclothes_max = 10\nseecharacterwearitem_max = 10\nseecharacteritem_max = 10\nfood_shop_item_max = 10\nfood_shop_type_max = 28\nfont = Sarasa Mono SC\nfont_size = 0\nnsfw = 1"
""" 配置文件信息 """

USER_PATH = ""
""" 用户数据路径 """
if platform.system() == "Linux":
    USER_PATH = appdirs.user_config_dir("dieloli")
else:
    USER_PATH = "."
os.makedirs(USER_PATH, exist_ok=True)

CONFIG_PATH = os.path.join(USER_PATH,"config.ini")
""" 配置文件路径 """
if not os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH,"w",encoding="utf-8") as config_file:
        config_file.write(config_info)

SAVE_PATH = os.path.join(USER_PATH, "save")
""" 存档目录路径 """
os.makedirs(SAVE_PATH, exist_ok=True)

MAP_DATA_PATH = os.path.join(USER_PATH, "data")
""" 地图预热数据路径 """
os.makedirs(MAP_DATA_PATH, exist_ok=True)
