from typing import List
from Script.Design import talk, talk_cache


"""
默认口上模板
"""


@talk.add_talk("default", "Rest")
def talk_rest() -> List[str]:
    """
    生成休息时口上
    """
    if talk_cache.me == talk_cache.tg:
        return [
            talk_cache.me.nick_name + "小小的休息了一会儿",
            talk_cache.me.nick_name + "长长的呼了一口气",
            talk_cache.me.nick_name + "打了个盹",
        ]
    if talk_cache.tg.intimate > 20000 and talk_cache.tg.sex == "Woman":
        talk_text = (
            talk_cache.tg.name + "将头靠在" + talk_cache.me.nick_name + "肩上"
        )
        if talk_cache.me.nature["Enthusiasm"] > 30:
            talk_text += (
                "\n"
                + talk_cache.me.nick_name
                + "露出微笑，轻轻揽住了"
                + talk_cache.tg.name
                + "的肩"
            )
        return [talk_text]
    if talk_cache.tg.intimate > 1000 and talk_cache.tg.sex == "Woman":
        return [
            talk_cache.tg.name
            + "将头靠在"
            + talk_cache.me.nick_name
            + "肩上小小的休息了一会儿"
        ]
    return ["与" + talk_cache.tg.name + "一起休息了一会儿"]
