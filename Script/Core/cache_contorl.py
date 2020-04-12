import concurrent.futures

# 流程用变量组
flow_contorl = {}

# 主页监听控制流程用变量组
wframe_mouse = {}

# cmd存储
cmd_map = {}

# 角色对象数据缓存组
character_data = {}

# npc模板列表
npc_tem_data = []

# 输入记录（最大20）
input_cache = []

now_init_map_id = ""

# 回溯输入记录用定位
input_position = {}

instruct_filter = {}

# 富文本记录输出样式临时缓存
output_text_style = ""

family_region_list = []
boys_region_list = []
girls_region_list = []
family_region_int_list = []
boys_region_int_list = []
girls_region_int_list = []

# 富文本回溯样式记录用定位
text_style_position = {}

# 存储服装类型数据
clothing_type_data = {}

# 富文本样式记录
text_style_cache = []

# 富文本精确样式记录
text_one_by_one_rich_cache = {}

last_cursor = [0]

# 图片id
image_id = 0

# cmd数据
cmd_data = {}

# 游戏时间
game_time = {}

# 时间增量
sub_game_time = 0

# 面板状态
panel_state = {}

# 存档页面最大数量
max_save_page = 0

# 现在所处的流程
now_flow_id = ""
old_flow_id = ""
too_old_flow_id = ""

# 课时数据
course_data = {}

# 教师科目经验
teacher_course_experience = {}

old_character_id = 0

# 各年龄段总人数
total_number_of_people_of_all_ages = {}

# 各年龄段总体脂率
total_bodyfat_by_age = {}

# 各年龄段平均体脂率
average_bodyfat_by_age = {}

# 各年龄段总身高
total_height_by_age = {}

# 各年龄段平均身高
average_height_by_age = {}

# 身材描述文本权重数据
stature_descrition_priorition_data = {}

text_wait = 0

map_data = {}
scene_data = {}

now_map = []

random_npc_list = []

occupation_character_data = {}

place_data = {}

# 可穿戴道具类型数据
wear_item_type_data = {}

# 当前上课时间状态
course_time_status = {}
