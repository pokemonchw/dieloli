from functools import wraps
from types import FunctionType
from Script.Core import cache_control, game_type, get_text, text_handle
from Script.Design import attr_text, constant
from Script.UI.Model import panel
from Script.Config import game_config, normal_config

cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """
width = 100
""" 屏幕宽度 """
_: FunctionType = get_text._
""" 翻译api """
describe_list = [_("可爱的"), _("性感的"), _("帅气的"), _("清新的"), _("典雅的"), _("清洁的"), _("保暖的")]


def handle_settle_behavior(character_id: int, now_time: int, event_id: str) -> panel.LeftDrawTextListWaitPanel:
    """
    处理结算角色事件
    Keyword arguments:
    character_id -- 角色id
    now_time -- 结算的时间戳
    event_id -- 触发的事件id
    Return arguments:
    panel.LeftDrawTextListWaitPanel -- 结算的文本列表
    self_draw_judge -- 结算的文本里是否有自身的信息
    """
    now_character_data: game_type.Character = cache.character_data[character_id]
    status_data = game_type.CharacterStatusChange()
    start_time = now_character_data.behavior.start_time
    add_time = int((now_time - start_time) / 60)
    event_data: game_type.Event = game_config.config_event[event_id]
    for settle in event_data.settle:
        try:
            constant.settle_behavior_effect_data[settle](
                character_id, add_time, status_data, now_time
            )
        except Exception as e:
            print(e)
            continue
    change_character_favorability_for_time(character_id, now_time)
    change_character_social(character_id, status_data)
    now_judge = False
    if status_data is None:
        return
    player_data: game_type.Character = cache.character_data[0]
    if (
        character_id
        and character_id != player_data.target_character_id
        and now_character_data.target_character_id
    ):
        return
    if status_data.mana_point:
        now_judge = 1
    if status_data.hit_point:
        now_judge = 1
    if status_data.knowledge:
        now_judge = 1
    if status_data.language:
        now_judge = 1
    if status_data.status:
        now_judge = 1
    if status_data.sex_experience:
        now_judge = 1
    if status_data.wear:
        now_judge = 1
    if status_data.undress:
        now_judge = 1
    if status_data.target_change and not character_id:
        now_judge = 1
    if now_judge:
        now_text_list = []
        if status_data.hit_point and round(status_data.hit_point, 2) != 0:
            now_text_list.append(
                _("健康:") + text_handle.number_to_symbol_string(round(status_data.hit_point, 2))
            )
        if status_data.mana_point and round(status_data.mana_point, 2) != 0:
            now_text_list.append(
                _("体力:") + text_handle.number_to_symbol_string(round(status_data.mana_point, 2))
            )
        if len(now_text_list) < 2:
            now_text_list.append("")
        if len(now_text_list):
            if len(now_text_list) % 4:
                now_text_list.extend([""] * (len(now_text_list) % 4))
        if status_data.status:
            now_list = [f"{game_config.config_character_state[i].name}:{attr_text.get_value_text(status_data.status[i])}" for i in status_data.status if round(status_data.status[i], 2)]
            if now_list:
                now_text_list.extend([_("状态变化:"), "", "", ""])
                now_text_list.extend(now_list)
        if len(now_text_list):
            if len(now_text_list) % 4:
                now_text_list.extend([""] * (len(now_text_list) % 4))
        if status_data.knowledge:
            now_list = [f"{game_config.config_knowledge[i].name}:{attr_text.get_value_text(status_data.knowledge[i])}" for i in status_data.knowledge if round(status_data.knowledge[i], 2)]
            if len(now_list):
                now_text_list.extend([_("增加技能经验:"), "", "", ""])
                now_text_list.extend(now_list)
        if len(now_text_list):
            if len(now_text_list) % 4:
                now_text_list.extend([""] * (len(now_text_list) % 4))
        if status_data.language:
            now_list = [f"{game_config.config_language[i].name}:{attr_text.get_value_text(status_data.language[i])}" for i in status_data.language if round(status_data.language[i], 2)]
            if len(now_list):
                now_text_list.extend([_("增加语言经验:"), "", "", ""])
                now_text_list.extend(now_list)
        if len(now_text_list):
            if len(now_text_list) % 4:
                now_text_list.extend([""] * (len(now_text_list) % 4))
        if status_data.sex_experience:
            now_list = [
                f"{game_config.config_organ[i].name}" + _("经验:") + text_handle.number_to_symbol_string(round(status_data.sex_experience[i], 2))
                for i in status_data.sex_experience if round(status_data.sex_experience[i], 2)
            ]
            if len(now_list):
                now_text_list.extend([_("增加性经验:"), "", "", ""])
                now_text_list.extend(now_list)
        if len(now_text_list):
            if len(now_text_list) % 4:
                now_text_list.extend([""] * (len(now_text_list) % 4))
        if status_data.wear:
            now_text_list.extend([_("穿上了:"), "", "", ""])
            for clothing in status_data.wear.values():
                clothing_config = game_config.config_clothing_tem[clothing.tem_id]
                value_list = [clothing.sweet,clothing.sexy,clothing.handsome,clothing.fresh,clothing.elegant,clothing.cleanliness,clothing.warm]
                describe_id = value_list.index(max(value_list))
                describe = describe_list[describe_id]
                clothing_name = f"[{clothing.evaluation}{describe}{clothing_config.name}]"
                now_text_list.append(clothing_name)
        if len(now_text_list):
            if len(now_text_list) % 4:
                now_text_list.extend([""] * (len(now_text_list) % 4))
        if status_data.undress:
            now_text_list.extend([_("脱下了:"), "", "", ""])
            for clothing in status_data.undress.values():
                clothing_config = game_config.config_clothing_tem[clothing.tem_id]
                value_list = [clothing.sweet,clothing.sexy,clothing.handsome,clothing.fresh,clothing.elegant,clothing.cleanliness,clothing.warm]
                describe_id = value_list.index(max(value_list))
                describe = describe_list[describe_id]
                clothing_name = f"[{clothing.evaluation}{describe}{clothing_config.name}]"
                now_text_list.append(clothing_name)
        if len(now_text_list):
            if len(now_text_list) % 4:
                now_text_list.extend([""] * (len(now_text_list) % 4))
        target_change_judge = False
        target_draw_list = []
        if status_data.target_change:
            for target_character_id in status_data.target_change:
                if character_id and target_character_id:
                    continue
                target_change: game_type.TargetChange = status_data.target_change[
                    target_character_id
                ]
                target_text_list = []
                target_data: game_type.Character = cache.character_data[target_character_id]
                target_text_list.extend(["-", "-", "-", "-"])
                target_text_list.extend([target_data.name, "", "", ""])
                target_text_judge = False
                if target_change.favorability:
                    favorability_text = _("好感:") + text_handle.number_to_symbol_string(round(target_change.favorability, 2))
                    social_type_change_text = ""
                    if target_change.favorability > 0:
                        social_type_change_text = get_social_type_change_text(target_data.favorability[character_id], False)
                    else:
                        social_type_change_text = get_social_type_change_text(target_data.favorability[character_id], True)
                    target_text_list.extend([favorability_text, social_type_change_text, "", ""])
                    target_text_judge = True
                if target_change.new_social != target_change.old_social:
                    now_text = game_config.config_social_type[target_change.old_social].name + "->" + game_config.config_social_type[target_change.new_social].name
                    target_text_list.extend([now_text, ""])
                    target_text_judge = True
                if target_change.status:
                    now_list = [f"{game_config.config_character_state[i].name}:{attr_text.get_value_text(target_change.status[i])}" for i in target_change.status if round(target_change.status[i], 2)]
                    if now_list:
                        target_text_list.extend([_("状态变化:"), "", "", ""])
                        target_text_list.extend(now_list)
                        target_text_judge = True
                if len(now_text_list):
                    if len(now_text_list) % 4:
                        target_text_list.extend([""] * (len(now_text_list) % 4))
                        target_text_judge = True
                if target_change.sex_experience:
                    now_list = [
                        f"{game_config.config_organ[i].name}" + _("经验:") + text_handle.number_to_symbol_string(round(target_change.sex_experience[i], 2))
                        for i in target_change.sex_experience if round(target_change.sex_experience[i], 2)
                    ]
                    if len(now_list):
                        target_text_list.extend([_("增加性经验:"), "", "", ""])
                        target_text_list.extend(now_list)
                        target_text_judge = True
                if target_text_judge:
                    target_draw_list.extend(target_text_list)
                    target_change_judge = True
        if target_change_judge:
            now_text_list.extend([_("互动目标变化:"), "", "", ""])
            now_text_list.extend(target_draw_list)
        now_panel = panel.LeftDrawTextListWaitPanel()
        now_panel.set(now_text_list, width, 4, True)
        return now_panel, len(now_text_list)


def add_settle_behavior_effect(behavior_effect_id: int):
    """
    添加行为结算处理
    Keyword arguments:
    behavior_id -- 行为id
    """

    def decorator(func):
        @wraps(func)
        def return_wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        constant.settle_behavior_effect_data[behavior_effect_id] = return_wrapper
        return return_wrapper

    return decorator


def get_cut_down_favorability_for_consume_time(consume_time: int):
    """
    从经过的时间计算出扣除的好感度
    Keyword arguments:
    consume_time -- 经过时间
    """
    if consume_time < 10:
        return consume_time
    if 10 <= consume_time < 100:
        return (consume_time - 9) * 10 + 9
    if 100 <= consume_time < 1000:
        return (consume_time - 99) * 100 + 909
    return (consume_time - 999) * 1000 + 90909


def change_character_favorability_for_time(character_id: int, now_time: int):
    """
    按最后社交时间扣除角色好感度
    Keyword arguments:
    character_id -- 角色id
    now_time -- 当前时间戳
    """
    character_data: game_type.Character = cache.character_data[character_id]
    for now_character in character_data.favorability:
        if character_data.favorability[now_character] > 0:
            character_data.social_contact_last_time.setdefault(now_character, now_time)
            last_add_time = character_data.social_contact_last_time[now_character]
            now_consume_time = int((now_time - last_add_time) / 60)
            if now_consume_time < 60:
                continue
            now_cut_down = get_cut_down_favorability_for_consume_time(int(now_consume_time / 60))
            if now_character in character_data.social_contact_last_cut_down_time:
                last_cut_down_time = character_data.social_contact_last_cut_down_time[now_character]
                old_consume_time = int((last_cut_down_time - last_add_time) / 60)
                old_cut_down = get_cut_down_favorability_for_consume_time(int(old_consume_time / 60))
                now_cut_down -= old_cut_down
            character_data.favorability[now_character] -= now_cut_down
            change_character_social_now(now_character, character_id)
        elif character_data.favorability[now_character] < 0:
            character_data.social_contact_last_time.setdefault(now_character, now_time)
            last_add_time = character_data.social_contact_last_time[now_character]
            now_consume_time = int((now_time - last_add_time) / 60)
            if now_consume_time < 60:
                continue
            now_cut_down = get_cut_down_favorability_for_consume_time(int(now_consume_time / 60))
            if now_character in character_data.social_contact_last_cut_down_time:
                last_cut_down_time = character_data.social_contact_last_cut_down_time[now_character]
                old_consume_time = int((last_cut_down_time - last_add_time) / 60)
                old_cut_down = get_cut_down_favorability_for_consume_time(int(old_consume_time / 60))
                now_cut_down -= old_cut_down
            character_data.favorability[now_character] += now_cut_down
            change_character_social_now(now_character, character_id)


def change_character_social(character_id: int, change_data: game_type.CharacterStatusChange):
    """
    处理角色关系变化
    Keyword argumenys:
    character_id -- 状态变化数据所属角色id
    change_data -- 状态变化数据
    """
    for now_character in change_data.target_change:
        change_character_social_now(character_id, now_character, change_data)


def change_character_social_now(
    character_id: int,
    target_id: int,
    change_data: game_type.CharacterStatusChange = game_type.CharacterStatusChange(),
):
    """
    执行角色关系变化
    Keyword arguments:
    character_id -- 状态变化数据所属角色id
    target_id -- 关系变化角色id
    change_data -- 状态变化数据
    """
    if target_id in change_data.target_change:
        target_change: game_type.TargetChange = change_data.target_change[target_id]
    target_data: game_type.Character = cache.character_data[target_id]
    old_social = 5
    if character_id in target_data.social_contact_data:
        old_social = target_data.social_contact_data[character_id]
    target_data.favorability.setdefault(character_id, 0)
    now_favorability = target_data.favorability[character_id]
    new_social = get_favorability_social(now_favorability)
    if new_social != old_social:
        if target_id in change_data.target_change:
            target_change.old_social = old_social
            target_change.new_social = new_social
        target_data.social_contact.setdefault(old_social, set())
        target_data.social_contact.setdefault(new_social, set())
        if character_id in target_data.social_contact[old_social]:
            target_data.social_contact[old_social].remove(character_id)
        target_data.social_contact[new_social].add(character_id)
        target_data.social_contact_data[character_id] = new_social


def get_favorability_social(favorability: int) -> int:
    """
    获取好感度对应社交关系
    Keyword arguments:
    favorability -- 好感度
    Return arguments:
    int -- 社交关系
    """
    if favorability < -20000:
        return 0
    if favorability < -10000:
        return 1
    if favorability < -5000:
        return 2
    if favorability < -2000:
        return 3
    if favorability < -1000:
        return 4
    if favorability < 1000:
        return 5
    if favorability < 2000:
        return 6
    if favorability < 5000:
        return 7
    if favorability < 10000:
        return 8
    if favorability < 20000:
        return 9
    return 10


def get_social_type_change_text(favorability: int, is_sub: bool) -> str:
    """
    获取社交关系变化进度文本
    Keyword arguments:
    favorability -- 好感度
    is_sub -- 是否是减少了好感
    Return arguments:
    str -- 变化文本
    """
    draw_text: str = _("距离{social_type}还需{need_favorability}点好感")
    now_social_type = get_favorability_social(favorability)
    if is_sub:
        if now_social_type == 0:
            draw_text = _("关系已经坏到了极点")
            return draw_text
        elif now_social_type == 1:
            social_type = game_config.config_social_type[0].name
            need_favorability = -20000 - favorability, 2
            draw_text = draw_text.format(social_type=social_type, need_favorability=round(need_favorability, 2))
            return draw_text
        elif now_social_type == 2:
            social_type = game_config.config_social_type[1].name
            need_favorability = -10000 - favorability, 2
            draw_text = draw_text.format(social_type=social_type, need_favorability=round(need_favorability, 2))
            return draw_text
        elif now_social_type == 3:
            social_type = game_config.config_social_type[2].name
            need_favorability = -5000 - favorability, 2
            draw_text = draw_text.format(social_type=social_type, need_favorability=round(need_favorability, 2))
            return draw_text
        elif now_social_type == 4:
            social_type = game_config.config_social_type[3].name
            need_favorability = -2000 - favorability, 2
            draw_text = draw_text.format(social_type=social_type, need_favorability=round(need_favorability, 2))
            return draw_text
        elif now_social_type == 5:
            social_type = game_config.config_social_type[4].name
            need_favorability = -1000 - favorability
            draw_text = draw_text.format(social_type=social_type, need_favorability=round(need_favorability, 2))
            return draw_text
        elif now_social_type == 6:
            social_type = game_config.config_social_type[5].name
            need_favorability = 1000 - favorability
            draw_text = draw_text.format(social_type=social_type, need_favorability=round(need_favorability, 2))
            return draw_text
        elif now_social_type == 7:
            social_type = game_config.config_social_type[6].name
            need_favorability = 2000 - favorability
            draw_text = draw_text.format(social_type=social_type, need_favorability=round(need_favorability, 2))
            return draw_text
        elif now_social_type == 8:
            social_type = game_config.config_social_type[7].name
            need_favorability = 5000 - favorability
            draw_text = draw_text.format(social_type=social_type, need_favorability=round(need_favorability, 2))
            return draw_text
        elif now_social_type == 9:
            social_type = game_config.config_social_type[8].name
            need_favorability = 10000 - favorability
            draw_text = draw_text.format(social_type=social_type, need_favorability=round(need_favorability, 2))
            return draw_text
        elif now_social_type == 10:
            social_type = game_config.config_social_type[9].name
            need_favorability = 20000 - favorability
            draw_text = draw_text.format(social_type=social_type, need_favorability=round(need_favorability, 2))
            return draw_text
    else:
        if now_social_type == 0:
            social_type = game_config.config_social_type[1].name
            need_favorability = -20000 - favorability
            draw_text = draw_text.format(social_type=social_type, need_favorability=round(need_favorability, 2))
            return draw_text
        elif now_social_type == 1:
            social_type = game_config.config_social_type[2].name
            need_favorability = -10000 - favorability
            draw_text = draw_text.format(social_type=social_type, need_favorability=round(need_favorability, 2))
            return draw_text
        elif now_social_type == 2:
            social_type = game_config.config_social_type[3].name
            need_favorability = -5000 - favorability
            draw_text = draw_text.format(social_type=social_type, need_favorability=round(need_favorability, 2))
            return draw_text
        elif now_social_type == 3:
            social_type = game_config.config_social_type[4].name
            need_favorability = -2000 - favorability
            draw_text = draw_text.format(social_type=social_type, need_favorability=round(need_favorability, 2))
            return draw_text
        elif now_social_type == 4:
            social_type = game_config.config_social_type[5].name
            need_favorability = -1000 - favorability
            draw_text = draw_text.format(social_type=social_type, need_favorability=round(need_favorability, 2))
            return draw_text
        elif now_social_type == 5:
            social_type = game_config.config_social_type[6].name
            need_favorability = 1000 - favorability
            draw_text = draw_text.format(social_type=social_type, need_favorability=round(need_favorability, 2))
            return draw_text
        elif now_social_type == 6:
            social_type = game_config.config_social_type[7].name
            need_favorability = 2000 - favorability
            draw_text = draw_text.format(social_type=social_type, need_favorability=round(need_favorability, 2))
            return draw_text
        elif now_social_type == 7:
            social_type = game_config.config_social_type[8].name
            need_favorability = 5000 - favorability
            draw_text = draw_text.format(social_type=social_type, need_favorability=round(need_favorability, 2))
            return draw_text
        elif now_social_type == 8:
            social_type = game_config.config_social_type[9].name
            need_favorability = 10000 - favorability
            draw_text = draw_text.format(social_type=social_type, need_favorability=round(need_favorability, 2))
            return draw_text
        elif now_social_type == 9:
            social_type = game_config.config_social_type[10].name
            need_favorability = 20000 - favorability
            draw_text = draw_text.format(social_type=social_type, need_favorability=round(need_favorability, 2))
            return draw_text
        elif now_social_type == 10:
            draw_text = _("爱情的极致")
            return draw_text
