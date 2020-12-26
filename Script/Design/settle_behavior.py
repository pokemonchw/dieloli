import datetime
from functools import wraps
from types import FunctionType
from Script.Core import cache_control, constant, game_type, get_text, text_handle
from Script.Design import game_time, talk, map_handle
from Script.UI.Moudle import panel, draw
from Script.Config import game_config, normal_config

cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """
width = normal_config.config_normal.text_width
""" 屏幕宽度 """
_: FunctionType = get_text._
""" 翻译api """


def handle_settle_behavior(character_id: int, now_time: datetime.datetime):
    """
    处理结算角色行为
    Keyword arguments:
    character_id -- 角色id
    now_time -- 结算时间
    """
    status_data: game_type.CharacterStatusChange = cache.settle_behavior_data[
        cache.character_data[character_id].behavior.behavior_id
    ](character_id, now_time)
    now_judge = False
    if status_data == None:
        return
    if status_data.mana_point:
        now_judge = True
    if status_data.hit_point:
        now_judge = True
    if len(status_data.knowledge):
        now_judge = True
    if len(status_data.language):
        now_judge = True
    if len(status_data.status):
        now_judge = True
    if len(status_data.favorability) and not character_id:
        now_judge = True
    if now_judge:
        if not character_id or character_id == cache.character_data[0].target_character_id:
            now_character_data = cache.character_data[character_id]
            now_draw = draw.NormalDraw()
            now_draw.text = "\n" + now_character_data.name + "\n"
            now_draw.width = width
            now_draw.draw()
            now_text_list = []
            if status_data.hit_point:
                now_text_list.append(_("体力:") + str(round(status_data.hit_point, 2)))
            if status_data.mana_point:
                now_text_list.append(_("气力:") + str(round(status_data.mana_point, 2)))
            if len(status_data.status):
                now_text_list.extend(
                    [
                        f"{game_config.config_character_state[i].name}:{round(status_data.status[i],2)}"
                        for i in status_data.status
                    ]
                )
            if len(status_data.knowledge):
                now_text_list.extend(
                    [
                        f"{game_config.config_knowledge[i].name}:{round(status_data.knowledge[i],2)}"
                        for i in status_data.knowledge
                    ]
                )
            if len(status_data.language):
                now_text_list.extend(
                    [
                        f"{game_config.config_language[i].name}:{round(status_data.language[i],2)}"
                        for i in status_data.language
                    ]
                )
            if len(status_data.favorability) and not character_id:
                now_text_list.extend(
                    [
                        _("{target_name}对{character_name}好感").format(
                            target_name=cache.character_data[i].name, character_name=now_character_data.name
                        )
                        + text_handle.number_to_symbol_string(round(status_data.favorability[i], 2))
                        for i in status_data.favorability
                    ]
                )
            now_panel = panel.LeftDrawTextListPanel()
            now_panel.set(now_text_list, width, 8)
            now_panel.draw()
            wait_draw = draw.WaitDraw()
            wait_draw.draw()


def add_settle_behavior(behavior_id: int):
    """
    添加行为结算处理
    Keyword arguments:
    behavior_id -- 行为id
    """

    def decorator(func):
        @wraps(func)
        def return_wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        cache.settle_behavior_data[behavior_id] = return_wrapper
        return return_wrapper

    return decorator
