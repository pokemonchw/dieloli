from typing import Dict, Set
from game_type import ClubData, Signal

club_list_data: Dict[str, ClubData] = {}
""" 全部社团数据 {社团唯一id:社团数据} """
now_file_path: str = ""
""" 当前数据文件路径 """
update_signal = Signal()
""" ui刷新信号 """
update_premise_signal = Signal()
""" 设置的社团门槛刷新信号 """
now_club_id: str = ""
""" 当前选中的社团id """
club_theme: Dict[str, str] = {}
""" 社团主题配置 {主题id:主题描述} """
club_theme_data: Dict[str, str] = {}
""" 社团主题配置数据 {主题描述:主题id} """
club_activity: Dict[str, str] = {}
""" 社团活动配置 {活动id:活动描述} """
club_activity_data: Dict[str, str] = {}
""" 社团活动配置数据 {活动描述:活动id} """
premise_data: Dict[str, str] = {}
""" 前提列表 """
premise_type_data: Dict[str, Set] = {}
""" 前提类型列表 """
