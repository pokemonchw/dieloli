from typing import List
from Script.Core import constant
from Script.Design import talk


@talk.add_talk(constant.Behavior.MOVE, 1, {"IsPlayer"})
def talk_move_1() -> List[str]:
    """
    玩家限定的基础移动口上
    Return arguments:
    list -- 返回的口上列表
    """
    return [
        "{NickName}来到了{SceneName}",
        "{NickName}移动到了{SceneName}",
        "{NickName}走到了{SceneName}",
    ]


@talk.add_talk(constant.Behavior.MOVE, 2, {"NoPlayer", "InPlayerScene"})
def talk_move_2() -> List[str]:
    """
    npc移动至玩家所在场景时的基础口上
    Return arguments:
    list -- 返回的口上列表
    """
    return [
        "{Name}来到了{SceneName}",
        "{Name}走了过来",
        "{Name}冲着{PlayerNickName}打了个招呼",
    ]


@talk.add_talk(constant.Behavior.MOVE, 3, {"NoPlayer", "LeavePlayerScene"})
def talk_move_3() -> List[str]:
    """
    npc离开玩家所在场景时的基础口上
    Return arguments:
    list -- 返回的口上列表
    """
    return [
        "{Name}朝着{SceneName}离开了",
        "{Name}走向了{SceneName}",
        "{Name}去了{SceneName}",
        "{Name}前往了{SceneName}",
    ]


@talk.add_talk(
    constant.Behavior.MOVE, 4, {"NoPlayer", "InPlayerScene", "PlayerIsAdore", "IsWoman", "IsStudent"},
)
def talk_move_4() -> List[str]:
    """
    爱慕玩家的萝莉npc移动至玩家所在场景时的口上
    Return arguments:
    list -- 返回的口上列表
    """
    talk_1 = "{Name}来到了{SceneName}\n"
    talk_1 += "她看见{PlayerNickName}的第一时间便扑了过来\n"
    talk_1 += "{Name}紧紧的抱住了{PlayerNickName},小脸不停的蹭着{PlayerNickName}的胸"
    return [talk_1]
