from types import FunctionType
from Script.Design import handle_adv, constant, handle_premise
from Script.Core import get_text, game_type, cache_control

_: FunctionType = get_text._
""" 翻译api """
cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """


@handle_adv.add_adv_handler(
    constant.AdvNpc.LI_WEI,
    _("李薇")
)
def handle_li_wei(character_id: int) -> bool:
    """
    验证角色是否符合李薇的创建条件
    李薇设定:
        性别:女
        年龄:12
        性格:文静，外冷内热
        擅长:文学，诗歌，写作
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    bool -- 验证结果
    """
    character_data: game_type.Character = cache.character_data[character_data]
    # 女性
    if character_data.sex != 1:
        return False
    # 12岁
    if character_data.age != 12:
        return False
    # 低调
    if not handle_premise.handle_premise(
        constant.Premise.IS_LOW_KEY,
        character_id
    ):
        return False
    # 孤僻
    if not handle_premise.handle_premise(
        constant.Premise.IS_SOLITARY,
        character_id
    ):
        return False
    # 乐观
    if not handle_premise.handle_premise(
        constant.Premise.IS_OPTIMISTIC,
        character_id
    ):
        return False
    # 守信
    if not handle_premise.handle_premise(
        constant.Premise.IS_KEEP_PROMISES,
        character_id
    ):
        return False
    # 无私
    if not handle_premise.handle_premise(
        constant.Premise.IS_SELFLESS,
        character_id
    ):
        return False
    # 薄情
    if not handle_premise.handle_premise(
        constant.Premise.IS_UNGRATEFUL,
        character_id
    ):
        return False
    # 严谨
    if not handle_premise.handle_premise(
        constant.Premise.IS_RIGOROUS,
        character_id
    ):
        return False
    # 自律
    if not handle_premise.handle_premise(
        constant.Premise.IS_AUTONOMY,
        character_id
    ):
        return False
    # 沉稳
    if not handle_premise.handle_premise(
        constant.Premise.IS_STEADY,
        character_id
    ):
        return False
    # 犹豫
    if not handle_premise.handle_premise(
        constant.Premise.IS_HESITATE,
        character_id
    ):
        return False
    # 坚韧
    if not handle_premise.handle_premise(
        constant.Premise.IS_TENACITY,
        character_id
    ):
        return False
    # 机敏
    if not handle_premise.handle_premise(
        constant.Premise.IS_ASTUTE,
        character_id
    ):
        return False
    # 浮躁
    if not handle_premise.handle_premise(
        constant.Premise.IS_IMPETUOUS,
        character_id
    ):
        return False
    # 阴险
    if not handle_premise.handle_premise(
        constant.Premise.IS_INSIDIOUS,
        character_id
    ):
        return False
    # 狭隘
    if not handle_premise.handle_premise(
        constant.Premise.IS_NARROW,
        character_id
    ):
        return False
    # 冷漠
    if not handle_premise.handle_premise(
        constant.Premise.IS_APATHY,
        character_id
    ):
        return False
    # 自卑
    if not handle_premise.handle_premise(
        constant.Premise.IS_INFERIORITY,
        character_id
    ):
        return False
    # 热衷
    if not handle_premise.handle_premise(
        constant.Premise.IS_KEEN,
        character_id
    ):
        return False
    # 文学天赋
    if not handle_premise.handle_premise(
        constant.Premise.LITERATURE_INTEREST_IS_HEIGHT,
        character_id
    ):
        return False
    # 诗歌天赋
    if not handle_premise.handle_premise(
        constant.Premise.POETRY_INTEREST_IS_HEIGHT,
        character_id
    ):
        return False
    # 写作天赋
    if not handle_premise.handle_premise(
        constant.Premise.WRITE_SKILLS_INTEREST_IS_HEIGHT,
        character_id
    ):
        return False
    return True


@handle_adv.add_adv_handler(
    constant.AdvNpc.LI_XUE,
    _("李雪")
)
def handle_li_xue(character_id: int) -> bool:
    """
    验证角色是否符合李雪的创建条件
    李雪设定:
        性别:女
        年龄:12
        性格:活泼，热情，阳光，积极，精灵古怪，调皮捣蛋
        擅长:运动
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    bool -- 验证结果
    """
    character_data: game_type.Character = cache.character_data[character_data]
    # 女性
    if character_data.sex != 1:
        return False
    # 11岁
    if character_data.age != 11:
        return False
    # 活泼
    if not handle_premise.handle_premise(
        constant.Premise.IS_LIVELY,
        character_id
    ):
        return False
    # 合群
    if not handle_premise.handle_premise(
        constant.Premise.IS_GREGARIOUS,
        character_id
    ):
        return False
    # 乐观
    if not handle_premise.handle_premise(
        constant.Premise.IS_OPTIMISTIC,
        character_id
    ):
        return False
    # 狡诈
    if not handle_premise.handle_premise(
        constant.Premise.IS_DECEITFUL,
        character_id
    ):
        return False
    # 无私
    if not handle_premise.handle_premise(
        constant.Premise.IS_SELFLESS,
        character_id
    ):
        return False
    # 重情
    if not handle_premise.handle_premise(
        constant.Premise.IS_HEAVY_FEELING,
        character_id
    ):
        return False
    # 松散
    if not handle_premise.handle_premise(
        constant.Premise.IS_RELAX,
        character_id
    ):
        return False
    # 自律
    if not handle_premise.handle_premise(
        constant.Premise.IS_AUTONOMY,
        character_id
    ):
        return False
    # 稚拙
    if not handle_premise.handle_premise(
        constant.Premise.IS_CHILDISH,
        character_id
    ):
        return False
    # 决断
    if not handle_premise.handle_premise(
        constant.Premise.IS_RESOLUTION,
        character_id
    ):
        return False
    # 坚韧
    if not handle_premise.handle_premise(
        constant.Premise.IS_TENACITY,
        character_id
    ):
        return False
    # 机敏 
    if not handle_premise.handle_premise(
        constant.Premise.IS_ASTUTE,
        character_id
    ):
        return False
    # 耐性
    if not handle_premise.handle_premise(
        constant.Premise.IS_TOLERANCE,
        character_id
    ):
        return False
    # 爽直
    if not handle_premise.handle_premise(
        constant.Premise.IS_STARAIGHTFORWARD,
        character_id
    ):
        return False
    # 宽和
    if not handle_premise.handle_premise(
        constant.Premise.IS_TOLERANT,
        character_id
    ):
        return False
    # 热情
    if not handle_premise.handle_premise(
        constant.Premise.IS_ENTHUSIASM,
        character_id
    ):
        return False
    # 自信
    if not handle_premise.handle_premise(
        constant.Premise.IS_SELF_CONFIDENCE,
        character_id
    ):
        return False
    # 热衷
    if not handle_premise.handle_premise(
        constant.Premise.IS_KEEN,
        character_id
    ):
        return False
    # 运动天赋
    if not handle_premise.handle_premise(
        constant.Premise.MOTION_SKILLS_INTEREST_IS_HEIGHT,
        character_id
    ):
        return False
    return True

