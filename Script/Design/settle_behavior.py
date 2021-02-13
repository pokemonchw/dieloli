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
    change_character_social(character_id, status_data)
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
    if len(status_data.social_change) and not character_id:
        now_judge = True
    if now_judge:
        character_judge = False
        now_character_data: game_type.Character = cache.character_data[character_id]
        if not character_id:
            character_judge = True
        else:
            player_data: game_type.Character = cache.character_data[0]
            if (
                character_id == player_data.target_character_id
                and now_character_data.position == player_data.position
            ):
                character_judge = True
        if character_judge:
            now_text_list = []
            now_draw = draw.NormalDraw()
            now_draw.text = "\n" + now_character_data.name + ": "
            now_draw.width = width
            now_draw.draw()
            if status_data.hit_point:
                now_text_list.append(_("体力:") + str(round(status_data.hit_point, 2)))
            if status_data.mana_point:
                now_text_list.append(_("气力:") + str(round(status_data.mana_point, 2)))
            if len(status_data.status):
                now_text_list.extend(
                    [
                        f"{game_config.config_character_state[i].name}:{text_handle.number_to_symbol_string(round(status_data.status[i],2))}"
                        for i in status_data.status
                    ]
                )
            if len(status_data.knowledge):
                now_text_list.extend(
                    [
                        f"{game_config.config_knowledge[i].name}:{text_handle.number_to_symbol_string(round(status_data.knowledge[i],2))}"
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
                        _("\n{target_name}对{character_name}好感").format(
                            target_name=cache.character_data[i].name, character_name=now_character_data.name
                        )
                        + text_handle.number_to_symbol_string(round(status_data.favorability[i], 2))
                        for i in status_data.favorability
                    ]
                )
            if len(status_data.social_change) and not character_id:
                now_text_list.extend(
                    [
                        _("\n{target_name}对{character_name}的感情变化:").format(
                            target_name=cache.character_data[i].name, character_name=now_character_data.name
                        )
                        + game_config.config_social_type[status_data.social_change[i].old_social].name
                        + "->"
                        + game_config.config_social_type[status_data.social_change[i].new_social].name
                        for i in status_data.social_change
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


def change_character_social(character_id: int, change_data: game_type.CharacterStatusChange):
    """
    处理角色关系变化
    Keyword argumenys:
    character_id -- 状态变化数据所属角色id
    change_data -- 状态变化数据
    """
    for now_character in change_data.favorability:
        now_character_data: game_type.Character = cache.character_data[now_character]
        old_social = 0
        new_social = 0
        if character_id in now_character_data.social_contact_data:
            old_social = now_character_data.social_contact_data[character_id]
        now_favorability = now_character_data.favorability[character_id]
        if now_favorability < 100:
            new_social = 0
        elif now_favorability < 1000:
            new_social = 1
        elif now_favorability < 2000:
            new_social = 2
        elif now_favorability < 5000:
            new_social = 3
        elif now_favorability < 10000:
            new_social = 4
        elif now_favorability >= 10000:
            new_social = 5
        if new_social != old_social:
            now_change = game_type.SocialChange()
            now_change.old_social = old_social
            now_change.new_social = new_social
            change_data.social_change[now_character] = now_change
            now_character_data.social_contact.setdefault(old_social, set())
            if character_id in now_character_data.social_contact[old_social]:
                now_character_data.social_contact[old_social].remove(character_id)
            now_character_data.social_contact.setdefault(new_social, set())
            now_character_data.social_contact[new_social].add(character_id)
            now_character_data.social_contact_data[character_id] = new_social
