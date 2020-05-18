from typing import List


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
        self.name = "主人公"
        """ 角色名字 """
        self.nick_name = "你"
        """ 他人对角色的称呼 """
        self.self_name = "我"
        """ 角色的自称 """
        self.species = "人类"
        """ 角色的种族 """
        self.sex = "Man"
        """ 角色性别 """
        self.age = 17
        """ 角色年龄 """
        self.end_age = 74
        """ 角色预期寿命 """
        self.intimate = 0
        """ 角色与玩家的亲密度 """
        self.graces = 0
        """ 角色的魅力值 """
        self.hit_point_max = 0
        """ 角色最大HP """
        self.hit_point = 0
        """ 角色当前HP """
        self.mana_point_max = 0
        """ 角色最大MP """
        self.mana_point = 0
        """ 角色当前MP """
        self.sex_experience = {}
        """ 角色的性经验数据 """
        self.sex_grade = {}
        """ 角色的性等级描述数据 """
        self.state = 0
        """ 角色当前状态 """
        self.engraving = {}
        """ 角色的刻印数据 """
        self.clothing = {
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
        self.item = {}
        """ 角色拥有的道具数据 """
        self.height = {}
        """ 角色的身高数据 """
        self.weight = {}
        """ 角色的体重数据 """
        self.measurements = {}
        """ 角色的三围数据 """
        self.behavior = {
            "StartTime": {},
            "Duration": 0,
            "BehaviorId": 0,
            "MoveTarget": [],
        }
        """ 角色当前行为状态数据 """
        self.gold = 0
        """ 角色所持金钱数据 """
        self.position = ["0"]
        """ 角色当前坐标数据 """
        self.classroom = []
        """ 角色所属班级坐标 """
        self.officeroom = ["0"]
        """ 角色所属办公室坐标 """
        self.knowledge = {}
        """ 角色知识技能等级数据 """
        self.language = {}
        """ 角色语言技能等级数据 """
        self.mother_tongue = "Chinese"
        """ 角色母语 """
        self.interest = {}
        """ 角色天赋数据 """
        self.dormitory = "0"
        """ 角色宿舍坐标 """
        self.birthday = {}
        """ 角色生日数据 """
        self.weigt_tem = "Ordinary"
        """ 角色体重模板 """
        self.bodyfat_tem = "Ordinary"
        """ 角色体脂率模板 """
        self.bodyfat = {}
        """ 角色体脂率数据 """
        self.sex_experience_tem = "None"
        """ 角色性经验模板 """
        self.clothing_tem = "Uniform"
        """ 角色生成服装模板 """
        self.chest_tem = "Ordinary"
        """ 角色罩杯模板 """
        self.chest = {}
        """ 角色罩杯数据 """
        self.nature = {}
        """ 角色性格数据 """
        self.status = {}
        """ 角色状态数据 """
        self.put_on = {}
        """ 角色已穿戴服装数据 """
        self.wear_item = {}
        """ 角色持有可穿戴道具数据 """
        self.hit_point_tem = "Ordinary"
        """ 角色HP模板 """
        self.mana_point_tem = "Ordinary"
        """ 角色MP模板 """
        self.social_contact = {}
        """ 角色社交关系数据 """
        self.occupation = ""
        """ 角色职业ID """


class TalkObject:
    """
    口上对象
    """

    def __init__(self, occupation: str, instruct: str, func: callable):
        """
        构造口上对象
        Keyword arguments:
        occupation -- 口上所属的职业
        instruct -- 口上对应的命令id
        func -- 生成口上的执行函数
        """
        self.occupation = occupation
        """ 口上所属的职业 """
        self.instruct = instruct
        """ 口上对应的命令id """
        self.func = func
        """ 生成口上的执行函数 """
