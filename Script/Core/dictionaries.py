from Script.Core import cache_contorl


def handle_text(text: str) -> str:
    """
    对文本中的宏进行转义处理
    Keyword arguments:
    string -- 需要进行转义处理的文本
    """
    if len(cache_contorl.character_data) > 0:
        character_id = cache_contorl.now_character_id
        character_name = cache_contorl.character_data[
            character_id
        ].name
        character_nick_name = cache_contorl.character_data[
            character_id
        ].nick_name
        character_self_name = cache_contorl.character_data[
            character_id
        ].self_name
        return text.format(
            Name=character_name,
            NickName=character_nick_name,
            SelfName=character_self_name,
        )
    return text
