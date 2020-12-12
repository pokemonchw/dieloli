from typing import List
from types import FunctionType
from Script.Core import constant, get_text
from Script.Design import talk

_: FunctionType = get_text._
""" 翻译api """


@talk.add_talk(constant.Behavior.EAT, 1, {constant.Premise.IS_PLAYER})
def talk_eat_1() -> List[str]:
    """
    玩家限定的基础进食口上
    Return arguments:
    list -- 返回的口上列表
    """
    return [
        _("{NickName}吃掉了{FoodName}"),
        _("{NickName}将{FoodName}吃掉了"),
        _("{NickName}吃了一些{FoodName}"),
    ]


@talk.add_talk(
    constant.Behavior.EAT,
    2,
    {constant.Premise.IS_PLAYER, constant.Premise.EAT_SPRING_FOOD},
)
def talk_eat_2() -> List[str]:
    """
    玩家食用春药品质的食物时的口上
    Return arguments:
    list -- 返回的口上列表
    """
    talk_1 = _("{NickName}吃了一些{FoodName}\n如春药一般的绝赞美味!\n{NickName}沉浸在了食物带来的快乐之中")
    return [talk_1]


@talk.add_talk(
    constant.Behavior.EAT,
    3,
    {constant.Premise.NO_PLAYER, constant.Premise.IN_PLAYER_SCENE},
)
def talk_eat_3() -> List[str]:
    """
    npc食用食物时的基础口上
    Return arguments:
    list -- 返回的口上列表
    """
    return [
        _("{Name}吃了一些{FoodName}"),
        _("{Name}将{FoodName}吃掉了"),
        _("{Name}吃掉了{FoodName}"),
    ]
