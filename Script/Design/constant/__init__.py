from types import FunctionType
from typing import List, Dict, Set
from Script.Design.constant.achieve import Achieve
from Script.Design.constant.adv_npc import AdvNpc
from Script.Design.constant.behavior import Behavior
from Script.Design.constant.behavior_effect import BehaviorEffect
from Script.Design.constant.character_status import CharacterStatus
from Script.Design.constant.dressing_style import DressingStyle
from Script.Design.constant.instruct import Instruct
from Script.Design.constant.instruct_type import InstructType
from Script.Design.constant.panel import Panel
from Script.Design.constant.premise import Premise
from Script.Design.constant.state_machine import StateMachine


i = 0
for k in Instruct.__dict__:
    if isinstance(Instruct.__dict__[k], int):
        setattr(Instruct, k, i)
        i += 1


handle_premise_data: Dict[str, FunctionType] = {}
""" 前提处理数据 """
handle_instruct_data: Dict[int, FunctionType] = {}
""" 指令处理数据 """
handle_instruct_name_data: Dict[int, str] = {}
""" 指令对应文本 """
instruct_type_data: Dict[int, Set] = {}
""" 指令类型拥有的指令集合 """
instruct_premise_data: Dict[int, Set] = {}
""" 指令显示的所需前提集合 """
instruct_to_type_data: Dict[int, int] = {}
""" 指令对应的指令类型数据 """
handle_state_machine_data: Dict[str, FunctionType] = {}
""" 角色状态机函数 """
family_region_list: Dict[int, str] = {}
""" 姓氏区间数据 """
boys_region_list: Dict[int, str] = {}
""" 男孩名字区间数据 """
girls_region_list: Dict[int, str] = {}
""" 女孩名字区间数据 """
family_region_int_list: List[int] = []
""" 姓氏权重区间数据 """
boys_region_int_list: List[int] = []
""" 男孩名字权重区间数据 """
girls_region_int_list: List[int] = []
""" 女孩名字权重区间数据 """
panel_data: Dict[int, FunctionType] = {}
"""
面板id对应的面板绘制函数集合
面板id:面板绘制函数对象
"""
place_data: Dict[str, List[str]] = {}
""" 按房间类型分类的场景列表 场景标签:场景路径列表 """
in_door_scene_list: List[str] = []
""" 室内场景列表 """
cmd_map: Dict[int, FunctionType] = {}
""" cmd存储 """
settle_behavior_effect_data: Dict[int, FunctionType] = {}
""" 角色行为结算处理器 处理器id:处理器 """
adv_name_data: Dict[str, str] = {}
""" 所有剧情npc名字数据 {adv id:npc名字} """
adv_name_set: Set[str] = set()
""" 所有剧情npc名字集合 """
