from Script.Core import cache_contorl, constant


@character_behavior.add_behavior("Student",constant.CharacterStatus.STATUS_ATTEND_CLASS)
def attend_class(character_id: int):
    """
    上课行为执行
    Keyword arguments:
    character_id -- 角色id
    """
    character_data = cache_contorl.character_data[character_id]
