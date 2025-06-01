import os
import platform
import appdirs
import shutil


config_info = "[game]\ngame_name = 死亡萝莉\nauthor = 任悠云\nverson_time = @2017-2024\nbackground = #000\nlanguage = zh_CN\nwindow_width = 1108\nwindow_hight = 1000\ntextbox_width = 203\ntextbox_hight = 50\ntext_width = 120\ntext_hight = 50\ninputbox_width = 133\nmax_save = 100\nsave_page = 10\ncharacterlist_show = 10\ntext_wait = 0\nhome_url = https://github.com/pokemonchw/dieloli\nlicenses_url = http://creativecommons.org/licenses/by-nc-sa/2.0/\nproportion_teacher = 1\nproportion_student = 23\nthreading_pool_max = 20\ninsceneseeplayer_max = 20\nseecharacterclothes_max = 10\nseecharacterwearitem_max = 10\nseecharacteritem_max = 10\nfood_shop_item_max = 10\nfood_shop_type_max = 28\nfont = Sarasa Mono SC\nfont_size = 0\nnsfw = 1\nai_mode = 0\nollama_mode = deepseek-r1:1.5b\nai_api_url =\nai_api_key =\nai_api_model =\nprompt = 请你扮演游戏叙述工具，但是你不在游戏中，请你仅以第二人称视角对游戏中角色的行动进行叙述，游戏中的人物[{npc_name}]进行了[{action_text}]，满足以下条件{premise_text_list},得到了行动开始事件[{start_text}]和行动结束事件[{end_text}]，请你写出并仅写出开始到结束之间的过渡文本让过程变得合理可信，文本需简洁明确且不包括开始事件和结束事件的文本内容，40字以内，请使用中文，请模仿原文的语言风格，请不要对描述进行格式化，请给出且只给出过渡文本的内容，不可以描述与[行动]的事情，不可以出现任何第一人称，[不可以出现(我)(我们)(角色)(交互对象)等词汇]，事件文本中的[你]指玩家，游戏是校园背景，请你避免对条件列表未涉及的内容作出假设(例如性别，喜好，身高，年龄，场景，时间，学历等)"
""" 配置文件信息 """

USER_PATH = ""
""" 用户数据路径 """
if platform.system() == "Linux":
    USER_PATH = appdirs.user_config_dir("dieloli")
else:
    USER_PATH = "."
os.makedirs(USER_PATH, exist_ok=True)

CONFIG_PATH = os.path.join(USER_PATH, "config.ini")
""" 配置文件路径 """
if not os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "w", encoding="utf-8") as config_file:
        config_file.write(config_info)

SAVE_PATH = os.path.join(USER_PATH, "save")
""" 存档目录路径 """
os.makedirs(SAVE_PATH, exist_ok=True)

MAP_DATA_PATH = os.path.join(USER_PATH, "data")
""" 地图预热数据路径 """
os.makedirs(MAP_DATA_PATH, exist_ok=True)

AI_MODEL_PATH = os.path.join(USER_PATH, "data", "policy_model.pth")
""" 模型文件路径 """
if not os.path.exists(AI_MODEL_PATH):
    src_ai_model_path = os.path.join("data", "policy_model.pth")
    shutil.copyfile(src_ai_model_path, AI_MODEL_PATH)
