from typing import List
from Script.Core import constant
from Script.Design import talk


@talk.add_talk(constant.Behavior.REST, 1, {constant.Premise.IS_PLAYER})
def talk_rest_1() -> List[str]:
    """
    玩家限定的基础休息口上
    Return arguments:
    list -- 返回的口上列表
    """
    return [
        "{NickName}小小的休息了一会儿",
        "{NickName}长长的呼了一口气",
        "{NickName}打了个盹",
        "{NickName}伸了个懒腰",
        "{NickName}咪起眼睛休息了一会儿",
    ]


@talk.add_talk(
    constant.Behavior.REST,
    2,
    {constant.Premise.NO_PLAYER, constant.Premise.IN_PLAYER_SCENE},
)
def talk_rest_2() -> List[str]:
    """
    npc和玩家处于同一场景时的基础休息口上
    Return arguments:
    list -- 返回的口上列表
    """
    return [
        "{Name}小小的休息了一会儿",
        "{Name}长长的呼了一口气",
        "{Name}打了个盹",
        "{Name}咪起眼睛休息了一会儿",
        "{Name}伸了个懒腰",
    ]


@talk.add_talk(
    constant.Behavior.REST,
    3,
    {constant.Premise.IS_PLAYER, constant.Premise.HAVE_TARGET},
)
def talk_rest_3() -> List[str]:
    """
    玩家和npc一起休息时的基础口上
    Return arguments:
    list -- 返回的口上列表
    """
    return [
        "{NickName}和{TargetName}一起休息了一会儿",
        "与{TargetName}一起休息了一会儿",
        "一起打了个盹",
    ]


@talk.add_talk(
    constant.Behavior.REST,
    4,
    {
        constant.Premise.NO_PLAYER,
        constant.Premise.TARGET_IS_ADORE,
        constant.Premise.TARGET_IS_PLAYER,
    },
)
def talk_rest_4() -> List[str]:
    """
    npc爱慕玩家并与玩家一起休息时的口上
    Return arguments:
    list -- 返回的口上列表
    """
    talk_1 = "{Name}将头靠在{PlayerNickName}肩上\n"
    talk_1 += "{PlayerNickName}露出微笑,轻轻揽住了{Name}的腰"
    talk_2 = "{PlayerNickName}从背后抱住了{Name}\n"
    talk_2 += "两人一起休息了一会儿"
    return [talk_1, talk_2]


@talk.add_talk(
    constant.Behavior.REST,
    5,
    {
        constant.Premise.NO_PLAYER,
        constant.Premise.TARGET_IS_ADMIRE,
        constant.Premise.TARGET_IS_PLAYER,
    },
)
def talk_rest_5() -> List[str]:
    """
    npc恋幕玩家并与玩家一起休息时的口上
    Return arguments:
    list -- 返回的口上列表
    """
    talk_1 = "{Name}将头靠在{PlayerNickName}肩上小小的休息了一会儿"
    return [talk_1]
