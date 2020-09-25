class SexTem:
    """ 性别对应描述和性别器官模板 """

    cid: int
    """ 性别id """
    name: str
    """ 性别名称 """
    has_man_organ: bool
    """ 是否有男性器官 """
    has_woman_organ: bool
    """ 是否有女性器官 """


class BarConfig:
    """ 比例条名字对应的状态图片和绘制宽度 """

    cid: int
    """ 比例条id """
    name: str
    """ 比例条名字 """
    ture_bar: str
    """ 进度条图片 """
    null_bar: str
    """ 背景条图片 """
    width: int
    """ 图片绘制宽度 """


class Organ:
    """ 器官对应性别限定和文字描述 """

    cid: int
    """ 器官id """
    organ_type: int
    """ 类型(0:女,1:男,2:通用) """
    name: str
    """ 名字 """


class AgeJudgeSexExperienceTem:
    """ 不同性别不同年龄段对应生成不同性经验模板的权重 """

    cid: int
    """ 模板id """
    sex: int
    """ 性别类型 """
    age: int
    """ 年龄段 """
    sex_exp_tem: int
    """ 性经验模板 """
    weight: int
    """ 权重 """


class HitPointTem:
    """ hp模板对应平均值 """

    cid: int
    """ 模板id """
    max_value: int
    """ 最大值 """


class WeightTem:
    """ 体重模板对应体重范围 """

    cid: int
    """ 模板id """
    min_value: float
    """ 最小值 """
    max_value: float
    """ 最大值 """


class ChestTem:
    """ 罩杯对应范围和生成权重 """

    cid: int
    """ 模板id """
    min_value: float
    """ 最小值 """
    max_value: float
    """ 最大值 """
    weight: int
    """ 权重 """


class FontConfig:
    """ 字体样式配置数据(富文本用) """

    cid: int
    """ 样式id """
    name: str
    """ 字体名 """
    foreground: str
    """ 前景色 """
    background: str
    """ 背景色 """
    font: str
    """ 字体 """
    font_size: int
    """ 字体大小 """
    bold: bool
    """ 加粗 """
    underline: bool
    """ 下划线 """
    italic: bool
    """ 斜体 """
    selectbackground: str
    """ 选中时背景色 """
    info: str
    """ 备注 """


class ClothingCollocational:
    """ 搭配模板 """

    cid: int
    """ 模板id """
    clothing_tem_type: int
    """ 搭配所属服装种类 """
    clothing_type: int
    """ 当前服装类型 """
    clothing_tem: int
    """ 当前服装种类 """
    collocational: int
    """ 搭配限制0:不限制1:优先2:禁用3:仅穿或不穿4:必须5:必须为所有此项之一 """
    info: str
    """ 备注 """


class EndAgeTem:
    """ 性别对应平均寿命 """

    cid: int
    """ 模板id """
    sex: int
    """ 性别id """
    end_age: int
    """ 平均寿命 """


class ClothingTem:
    """ 服装模板 """

    cid: int
    """ 模板id """
    name: str
    """ 服装名字 """
    clothing_type: int
    """ 服装类型 """
    sex: int
    """ 服装性别限制 """
    tag: int
    """ 服装用途标签 """
    describe: str
    """ 描述 """


class FoodQualityWeight:
    """ 烹饪技能等级制造食物品质权重配置 """

    cid: int
    """ 配置表id """
    level: int
    """ 烹饪技能等级 """
    quality: int
    """ 食物品质 """
    weight: int
    """ 权重 """


class RandomNpcSexWeight:
    """ 生成随机npc时性别权重 """

    cid: int
    """ 模板id """
    sex: int
    """ 性别id """
    weight: int
    """ 权重 """


class ManaPointTem:
    """ mp模板对应平均值 """

    cid: int
    """ 模板id """
    max_value: int
    """ 最大值 """


class AgeTem:
    """ 不同年龄段模板的年龄范围 """

    cid: int
    """ 模板id """
    max_age: int
    """ 最大年龄 """
    min_age: int
    """ 最小年龄 """


class SexExperience:
    """ 性经验丰富程度模板对应器官性经验模板 """

    cid: int
    """ 表id """
    sex_exp_type: int
    """ 经验类型 """
    organ_id: int
    """ 器官id """
    exp_tem: int
    """ 经验模板 """


class AttrTem:
    """ 性别对应的角色各项基础属性模板 """

    cid: int
    """ 模板id """
    sex: int
    """ 性别 """
    age_tem: int
    """ 年龄模板id """
    hit_point_tem: int
    """ HP模板id """
    mana_point_tem: int
    """ MP模板id """
    sex_experience: int
    """ 性经验模板id """
    weight_tem: int
    """ 体重模板id """
    body_fat_tem: int
    """ 体脂率模板id """


class SexExperienceTem:
    """ 器官类型性经验丰富程度对应经验范围 """

    cid: int
    """ 模板id """
    sex_exp_tem_type: int
    """ 器官类型 """
    sub_type: int
    """ 子类型 """
    max_exp: int
    """ 最大经验 """
    min_exp: int
    """ 最小经验 """


class ClothingType:
    """ 衣服种类配置 """

    cid: int
    """ 类型id """
    name: str
    """ 类型名字 """


class ClothingUseType:
    """ 服装用途配置 """

    cid: int
    """ 用途id """
    name: str
    """ 用途名字 """


class BodyFatTem:
    """ 按性别划分的体脂率模板和范围 """

    cid: int
    """ 模板id """
    sex_type: int
    """ 性别类型 0:男,1:女 """
    sub_type: int
    """ 体脂率子类型 """
    min_value: float
    """ 最小值 """
    max_value: float
    """ 最大值 """
