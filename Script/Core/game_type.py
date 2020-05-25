from typing import List, Dict


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
        self.AdvNpc: bool = 0
        """ 剧情npc校验 """
        self.Weight: str = ""
        """ 体重模板 """
        self.BodyFat: str = ""
        """ 体脂率模板 """
        self.Chest: str = ""
        """ 罩杯模板 """
        self.MotherTongue: str = ""
        """ 母语 """


class Character:
    """ 角色数据结构体 """

    def __init__(self):
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
        self.clothing: dict = {
            "Coat": {},
            "Underwear": {},
            "Pants": {},
            "Skirt": {},
            "Shoes": {},
            "Socks": {},
            "Bra": {},
            "Underpants": {},
        }
        """ 角色拥有的服装数据 """
        self.item: dict = {}
        """ 角色拥有的道具数据 """
        self.height: dict = {}
        """ 角色的身高数据 """
        self.weight: dict = {}
        """ 角色的体重数据 """
        self.measurements: dict = {}
        """ 角色的三围数据 """
        self.behavior: dict = {
            "StartTime": {},
            "Duration": 0,
            "BehaviorId": 0,
            "MoveTarget": [],
        }
        """ 角色当前行为状态数据 """
        self.gold: int = 0
        """ 角色所持金钱数据 """
        self.position: List[str] = ["0"]
        """ 角色当前坐标数据 """
        self.classroom: str = ""
        """ 角色所属班级坐标 """
        self.officeroom: List[str] = ["0"]
        """ 角色所属办公室坐标 """
        self.course_table: Dict[int, Dict[int, str]] = {}
        """ 教室角色课程表数据 """
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
        self.birthday: dict = {}
        """ 角色生日数据 """
        self.weigt_tem: str = "Ordinary"
        """ 角色体重模板 """
        self.bodyfat_tem: str = "Ordinary"
        """ 角色体脂率模板 """
        self.bodyfat: dict = {}
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
        self.status: Dict[str, int] = {}
        """ 角色状态数据 """
        self.put_on: dict = {}
        """ 角色已穿戴服装数据 """
        self.wear_item: dict = {}
        """ 角色持有可穿戴道具数据 """
        self.hit_point_tem: str = "Ordinary"
        """ 角色HP模板 """
        self.mana_point_tem: str = "Ordinary"
        """ 角色MP模板 """
        self.social_contact: dict = {}
        """ 角色社交关系数据 """
        self.occupation: str = ""
        """ 角色职业ID """
