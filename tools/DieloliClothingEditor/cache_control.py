from typing import Dict
from game_type import Signal, ClothingTem, ClothingSuit

update_signal = Signal()
""" ui刷新信号 """
clothing_list_data: Dict[str, ClothingTem] = {}
""" 全部服装数据 {服装唯一id:服装数据} """
suit_list_data: Dict[str, ClothingSuit] = {}
""" 全部套装数据 {套装唯一id:套装数据} """
now_file_path: str = ""
""" 当前数据文件路径 """
now_suit_id: str = ""
""" 当前选中的套装id """
now_clothing_id: str = ""
""" 当前选中的服装id """
gender_type = {"男": 0, "女": 1, "通用": 2}
""" 性别数据 """
wear_type = {"外套": 0, "上衣": 1, "裤子": 2, "裙子": 3, "鞋子": 4, "袜子": 5, "胸罩": 6, "内裤": 7}
""" 穿戴位置设定 """
wear_type_data = {0: "外套", 1: "上衣", 2: "裤子", 3: "裙子", 4: "鞋子", 5: "袜子", 6: "胸罩", 7: "内裤"}
""" 穿戴位置数据 """
