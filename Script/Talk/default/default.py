from Script.Design import talk, talk_cache

@talk.add_talk("default","Rest")
def talk_rest():
    """
    生成休息时口上
    """
    if talk_cache.me == talk_cache.tg:
        return [
            talk_cache.me.self_name + "小小的休息了一会儿",
            talk_cache.me.self_name + "长长的呼了一口气",
            talk_cache.me.self_name + "打了个盹",
        ]
    if talk_cache.tg.name
    return [
        "与" + talk_cache.tg.name + "一起休息了一会儿"
    ]
