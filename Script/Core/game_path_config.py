import os
import platform

USER_PATH = "."
""" 当前用户路径 """

if platform.system() == "Linux":
    USER_PATH = os.path.join(os.path.expanduser("~"),".dieloli")

if not os.path.exists(USER_PATH):
    os.mkdir(USER_PATH)

SAVE_PATH = os.path.join(USER_PATH,"save")
""" 存档目录路径 """

if not os.path.exists(SAVE_PATH):
    os.mkdir(SAVE_PATH)

config_info = "[game]\ngame_name = 死亡萝莉\nauthor = 任悠云\nverson_time = @2017-2023\nbackground = #000\nlanguage = zh_CN\nwindow_width = 1108\nwindow_hight = 1000\ntextbox_width = 120\ntextbox_hight = 60\ntext_width = 120\ntext_hight = 50\ninputbox_width = 133\nyear = 2007\nmonth = 5\nday = 13\nhour = 6\nminute = 0\nmax_save = 100\nsave_page = 10\ncharacterlist_show = 10\ntext_wait = 0\nhome_url = https://github.com/pokemonchw/dieloli\nlicenses_url = http://creativecommons.org/licenses/by-nc-sa/2.0/\nrandom_npc_max = 2800\nproportion_teacher = 1\nproportion_student = 23\nthreading_pool_max = 20\ninsceneseeplayer_max = 20\nseecharacterclothes_max = 10\nseecharacterwearitem_max = 10\nseecharacteritem_max = 10\nfood_shop_item_max = 10\nfood_shop_type_max = 28\nfont = Sarasa Mono SC\nnsfw = 1"

CONFIG_PATH = os.path.join(USER_PATH,"config.ini")
""" 配置文件路径 """
if not os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH,"w",encoding="utf-8") as config_file:
        config_file.write(config_info)
