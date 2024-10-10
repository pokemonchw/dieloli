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
    character_data: game_type.Character = cache.character_data[character_id]
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
    # 文学天赋
    if not handle_premise.handle_premise(
        constant.Premise.LITERATURE_INTEREST_IS_HEIGHT,
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
    character_data: game_type.Character = cache.character_data[character_id]
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
    # 机敏 
    if not handle_premise.handle_premise(
        constant.Premise.IS_ASTUTE,
        character_id
    ):
        return False
    # 热情
    if not handle_premise.handle_premise(
        constant.Premise.IS_ENTHUSIASM,
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

