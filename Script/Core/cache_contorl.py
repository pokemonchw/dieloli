from typing import TYPE_CHECKING, List, Dict
from uuid import UUID
from Script.Core import game_type


flow_contorl: game_type.FlowContorl = game_type.FlowContorl()
""" 流程用变量组 """
wframe_mouse: game_type.WFrameMouse = game_type.WFrameMouse()
""" 主页监听控制流程用变量组 """
cmd_map: Dict[int, object] = {}
""" cmd存储 """
character_data: Dict[int, game_type.Character] = {}
""" 角色对象数据缓存组 """
now_character_id: int = 0
""" 当前正在交互的角色id """
npc_tem_data: List[game_type.NpcTem] = []
""" npc模板列表 """
input_cache: List[str] = []
""" 玩家指令输入记录（最大20）"""
now_init_map_id: str = ""
""" 寻路算法用,当前节点所属的地图的id """
input_position: int = 0
""" 回溯输入记录用定位 """
instruct_filter: Dict[str, int] = {}
""" 玩家操作指令面板指令过滤状态数据 """
output_text_style: str = ""
""" 富文本记录输出样式临时缓存 """
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
text_style_position: int = 0
""" 富文本回溯样式记录用定位 """
clothing_type_data: dict = {}
""" 存储服装类型数据 """
text_style_cache: List[str] = []
""" 富文本样式记录 """
text_one_by_one_rich_cache: dict = {}
""" 富文本精确样式记录 """
image_id: int = 0
""" 图片id """
cmd_data: dict = {}
""" cmd数据 """
game_time: dict = None
""" 游戏时间 """
panel_state: Dict[str, int] = {}
""" 面板状态 """
max_save_page: int = 0
""" 存档页面最大数量 """
now_flow_id: str = ""
""" 当前游戏控制流程id """
old_flow_id: str = ""
""" 上次游戏控制流程id """
too_old_flow_id: str = ""
""" 上上次游戏控制流程id """
course_data: dict = {}
""" 各个年级各科目课时数据 """
teacher_course_experience: dict = {}
""" 教师科目经验 """
old_character_id: int = 0
""" 离开场景面板前在场景中查看的角色id """
total_number_of_people_of_all_ages: Dict[str, int] = {}
""" 各年龄段总人数 """
total_bodyfat_by_age: dict = {}
""" 各年龄段总体脂率 """
average_bodyfat_by_age: dict = {}
""" 各年龄段平均体脂率 """
total_height_by_age: dict = {}
""" 各年龄段总身高 """
average_height_by_age: dict = {}
""" 各年龄段平均身高 """
stature_descrition_priorition_data: dict = {}
""" 身材描述文本权重数据 """
text_wait: int = 0
""" 绘制文本输出等待时间 """
map_data: dict = {}
""" 游戏地图数据 """
scene_data: dict = {}
""" 游戏场景数据 """
now_map: List[str] = []
""" 查看地图时当前所查看的地图的坐标 """
random_npc_list: List[game_type.NpcTem] = []
""" 随机npc数据 """
place_data: dict = {}
""" 按房间类型分类的场景列表 """
wear_item_type_data: dict = {}
""" 可穿戴道具类型数据 """
course_time_status: dict = {}
""" 当前上课时间状态 """
talk_data: dict = {}
""" 所有的口上数据 """
status_up_text: dict = {}
""" 显示给玩家的角色状态变化文本 """
behavior_tem_data: dict = {}
""" 角色行为控制器数据 """
settle_behavior_data: dict = {}
""" 角色行为结算处理器 """
over_behavior_character: Dict[int, int] = {}
""" 本次update中已结束结算的npc """
teacher_class_time_table: Dict[int, Dict[int, Dict[int, Dict[str, str]]]] = {}
"""
各班级各老师上课时间表
周几:上课时间:教师id:教室id:科目
"""
teacher_phase_table: Dict[int, int] = {}
""" 各老师所在年级 """
recipe_data: Dict[int, game_type.Recipes] = {}
""" 菜谱数据 """
restaurant_data: Dict[str, Dict[UUID, game_type.Food]] = {}
"""
食堂内贩卖的食物数据
食物名字:食物唯一id:食物对象
"""
handle_premise_data: Dict[str, callable] = {}
""" 前提处理数据 """
handle_target_data: Dict[str, callable] = {}
""" 目标类型数据 """
premise_target_table: Dict[str, set] = {}
"""
目标对应的所需前提集合
目标id:前提集合
"""
effect_target_table: Dict[str, set] = {}
"""
效果对应所需目标集合
效果id:目标集合
"""
