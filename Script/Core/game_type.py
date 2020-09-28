from uuid import UUID
from typing import List, Dict
import datetime


class FlowContorl:
    """ 流程控制用结构体 """

    restart_game: bool = 0
    """ 重启游戏 """
    quit_game: bool = 0
    """ 退出游戏 """


class WFrameMouse:
    """ 鼠标状态结构体 """

    w_frame_up: int = 2
    """ 逐字输出状态 """
    mouse_right: int = 0
    """ 鼠标右键按下 """
    w_frame_lines_up: int = 2
    """ 逐行输出状态 """
    mouse_leave_cmd: int = 1
    """ 鼠标左键事件 """
    w_frame_re_print: int = 0
    """ 再次载入游戏界面 """
    w_frame_lines_state: int = 2
    """ 逐行输出状态 """
    w_frame_mouse_next_line: int = 0
    """ 等待玩家确认后逐行 """


class NpcTem:
    """ npc模板用结构体对象 """

    def __init__(self):
        self.Name: str = ""
        """ npc名字 """
        self.Sex: str = ""
        """ npc性别 """
        self.Age: str = ""
        """ npc年龄模板 """
        self.Position: List[int] = []
        """ npc出生位置(已废弃) """
        self.AdvNpc: int = 0
        """ 剧情npc校验 """
        self.Weight: str = ""
        """ 体重模板 """
        self.BodyFat: str = ""
        """ 体脂率模板 """
        self.Chest: int = 0
        """ 罩杯模板 """
        self.MotherTongue: str = ""
        """ 母语 """
        self.SexExperienceTem: str = ""
        """ 性经验模板 """


class Character:
    """ 角色数据结构体 """

    def __init__(self):
        self.id: int = 0
        """ 角色id """
        self.name: str = "主人公"
        """ 角色名字 """
        self.nick_name: str = "你"
        """ 他人对角色的称呼 """
        self.self_name: str = "我"
        """ 角色的自称 """
        self.species: str = "人类"
        """ 角色的种族 """
        self.sex: str = "Man"
        """ 角色性别 """
        self.age: int = 17
        """ 角色年龄 """
        self.end_age: int = 74
        """ 角色预期寿命 """
        self.intimate: int = 0
        """ 角色与玩家的亲密度 """
        self.graces: int = 0
        """ 角色的魅力值 """
        self.hit_point_max: int = 0
        """ 角色最大HP """
        self.hit_point: int = 0
        """ 角色当前HP """
        self.mana_point_max: int = 0
        """ 角色最大MP """
        self.mana_point: int = 0
        """ 角色当前MP """
        self.sex_experience: Dict[str, int] = {}
        """ 角色的性经验数据 """
        self.sex_grade: Dict[str, str] = {}
        """ 角色的性等级描述数据 """
        self.state: int = 0
        """ 角色当前状态 """
        self.engraving: Dict[str, int] = {}
        """ 角色的刻印数据 """
        self.clothing: Dict[int,Dict[UUID,Clothing]] = {}
        """
        角色拥有的服装数据
        服装穿戴位置:服装唯一id:服装数据
        """
        self.item: dict = {}
        """ 角色拥有的道具数据 """
        self.height: Height = Height()
        """ 角色的身高数据 """
        self.weight: dict = {}
        """ 角色的体重数据 """
        self.measurements: dict = {}
        """ 角色的三围数据 """
        self.behavior: Behavior = Behavior()
        """ 角色当前行为状态数据 """
        self.gold: int = 0
        """ 角色所持金钱数据 """
        self.position: List[str] = ["0"]
        """ 角色当前坐标数据 """
        self.classroom: str = ""
        """ 角色所属班级坐标 """
        self.officeroom: List[str] = ""
        """ 角色所属办公室坐标 """
        self.knowledge: Dict[str, int] = {}
        """ 角色知识技能等级数据 """
        self.language: Dict[str, int] = {}
        """ 角色语言技能等级数据 """
        self.mother_tongue: str = "Chinese"
        """ 角色母语 """
        self.interest: Dict[str, int] = {}
        """ 角色天赋数据 """
        self.dormitory: str = "0"
        """ 角色宿舍坐标 """
        self.birthday: datetime.datetime = datetime.datetime(1, 1, 1)
        """ 角色生日数据 """
        self.weigt_tem: str = "Ordinary"
        """ 角色体重模板 """
        self.bodyfat_tem: str = "Ordinary"
        """ 角色体脂率模板 """
        self.bodyfat: int = 0
        """ 角色体脂率数据 """
        self.sex_experience_tem: str = "None"
        """ 角色性经验模板 """
        self.clothing_tem: str = "Uniform"
        """ 角色生成服装模板 """
        self.chest_tem: str = "Ordinary"
        """ 角色罩杯模板 """
        self.chest: dict = {}
        """ 角色罩杯数据 """
        self.nature = {}
        """ 角色性格数据 """
        self.status: Dict[int, Dict[int, int]] = {}
        """ 角色状态数据 状态类型:状态id:状态数值 """
        self.put_on: dict = {}
        """ 角色已穿戴服装数据 """
        self.wear_item: dict = {}
        """ 角色持有可穿戴道具数据 """
        self.hit_point_tem: str = "Ordinary"
        """ 角色HP模板 """
        self.mana_point_tem: str = "Ordinary"
        """ 角色MP模板 """
        self.social_contact: dict = {
            "Adore": {},
            "Admire": {},
            "Subside": {},
            "Depend": {},
            "Trust": {},
            "Friendly": {},
        }
        """ 角色社交关系数据 关系id:角色id:亲密度 """
        self.occupation: str = ""
        """ 角色职业ID """
        self.food_bag: Dict[UUID, Food] = {}
        """ 角色持有的食物数据 """
        self.course: CourseTimeSlice = CourseTimeSlice()
        """ 上课时间和状态数据 """
        self.target_character_id: int = 0
        """ 角色当前交互对象id """
        self.adv: int = 0
        """ 剧情npc校验 """


class Food:
    """ 食物数据结构体 """

    def __init__(self):
        self.id: str = ""
        """ 食物配置表id """
        self.uid: UUID = None
        """ 食物对象的唯一id """
        self.quality: int = 0
        """ 食物品质 """
        self.weight: int = 0
        """ 食物重量 """
        self.feel: dict = {}
        """ 食物效果 """
        self.maker: str = ""
        """ 食物制作者 """
        self.recipe: int = -1
        """ 食谱id """
        self.cook: bool = False
        """ 可烹饪 """
        self.eat: bool = False
        """ 可食用 """
        self.seasoning: bool = False
        """ 可作为调料 """
        self.fruit: bool = False
        """ 是否是水果 """


class Recipes:
    """ 菜谱数据结构体 """

    def __init__(self):
        self.name: str = ""
        """ 菜谱名字 """
        self.time: int = 0
        """ 标准烹饪时间 """
        self.base: list = []
        """ 烹饪所使用的主食材 """
        self.ingredients: list = []
        """ 烹饪所使用的辅食材 """
        self.seasoning: list = []
        """ 烹饪所使用的调料 """


class CourseTimeSlice:
    """ 上课时间和状态数据结构体 """

    def __init__(self):
        self.in_course: bool = 0
        """ 当前时间是否是上课时间 """
        self.to_course: int = 0
        """ 当前距离下节课开始所需的时间 """
        self.course_index: int = 0
        """ 当前属于第几节课 """
        self.end_course: int = 0
        """ 当前距离下课时间所需的时间 """
        self.course_id: str = ""
        """ 课目id """
        self.school_id: str = ""
        """ 学校id """
        self.phase: int = 0
        """ 年级编号 """


class NormalConfig:
    """ 通用配置 """

    game_name: str
    """ 游戏名 """
    verson:str
    """ 游戏版本号 """
    author:str
    """ 作者名 """
    verson_time:str
    """ 版本时间 """
    background:str
    """ 背景色 """
    language:str
    """ 语言 """
    window_width:int
    """ 窗体宽度 """
    window_hight:int
    """ 窗体高度 """
    textbox_width:int
    """ 文本框字符宽度 """
    textbox_hight:int
    """ 文本框字符高度 """
    text_width:int
    """ 绘制用单行文本宽度 """
    text_hight:int
    """ 绘制用单屏行数 """
    inputbox_width:int
    """ 输入框宽度 """
    year:int
    """ 游戏时间开始年份 """
    month:int
    """ 游戏时间开始月份 """
    day:int
    """ 游戏时间开始日期 """
    hour:int
    """ 游戏时间开始小时数 """
    minute:int
    """ 游戏时间开始分钟数 """
    max_save:int
    """ 游戏存档数量上限 """
    save_page:int
    """ 存档显示页面单页存档数 """
    characterlist_show:int
    """ 角色列表单页显示角色数 """
    text_wait:int
    """ 步进文本等待时间 """
    home_url:str
    """ 开发者主页链接 """
    licenses_url:str
    """ 知识产权共享协议链接 """
    random_npc_max:int
    """ 最大随机npc数量 """
    proportion_teacher:int
    """ 生成教师权重 """
    proportion_student:int
    """ 生成学生权重 """
    insceneseeplayer_max:int
    """ 场景单页显示角色数上限 """
    seecharacterclothes_max:int
    """ 角色服装列表单页显示服装数上限 """
    seecharacterwearitem_max:int
    """ 角色可穿戴道具列表单页显示上限 """
    seecharacteritem_max:int
    """ 角色背包单页显示道具数上限 """
    food_shop_item_max:int
    """ 食物商店单页显示道具数上限 """
    food_shop_type_max:int
    """ 食物商店单页显示食物种类数上限 """


class Clothing:
    """ 服装数据结构体 """

    def __init__(self):
        self.uid:UUID = ""
        """ 服装对象的唯一id """
        self.tem_id:int = 0
        """ 服装配表id """
        self.sexy:int = 0
        """ 服装性感属性 """
        self.handsome:int = 0
        """ 服装帅气属性 """
        self.elegant:int = 0
        """ 服装典雅属性 """
        self.fresh:int = 0
        """ 服装清新属性 """
        self.sweet:int = 0
        """ 服装可爱属性 """
        self.warm:int = 0
        """ 服装保暖属性 """
        self.cleanliness:int = 0
        """ 服装清洁属性 """
        self.price:int = 0
        """ 服装价值属性 """
        self.evaluation:str = ""
        """ 服装评价文本 """
        self.wear:int = 0
        """ 穿戴部位 """


class Height:
    """ 身高数据结构体 """

    def __init__(self):
        self.now_height:int = 0
        """ 当前身高 """
        self.growth_height:int = 0
        """ 每日身高增量 """
        self.expect_age:int = 0
        """ 预期结束身高增长年龄 """
        self.development_age:int = 0
        """ 预期发育期结束时间 """
        self.expect_height:int = 0
        """ 预期的最终身高 """


class Behavior:
    """ 角色行为状态数据 """

    def __init__(self):
        self.start_time: datetime.datetime = None
        """ 行为开始时间 """
        self.duration:int = 0
        """ 行为持续时间(单位分钟) """
        self.behavior_id:int = 0
        """ 行为id """
        self.move_target:List[str] = []
        """ 移动行为目标坐标 """
        self.eat_food:Food = None
        """ 进食行为消耗的食物对象 """
        self.food_name:str = ""
        """ 前提结算用:进食行为消耗的食物名字 """
        self.food_quality:int = 0
        """ 前提结算用:进食行为消耗的食物品质 """
