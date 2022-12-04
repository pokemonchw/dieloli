from typing import Dict,Set
from game_type import Target

premise_data: Dict[str, str] = {}
""" 前提列表 """
premise_type_data: Dict[str, Set] = {}
""" 前提类型列表 """
state_machine_data: Dict[str, str] = {}
""" 状态机列表 """
state_machine_type_data: Dict[str, Set] = {}
""" 状态机类型列表 """
now_file_path: str = ""
""" 当前文件路径 """
now_target_data: Dict[str, Target] = {}
""" 当前ai数据 """
now_state_machine_id: str = ""
""" 当前选中的状态机id """
now_target_id: str = ""
""" 当前选中目标id """
