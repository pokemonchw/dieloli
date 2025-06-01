import random
import datetime
import re
from typing import Set, Dict
from types import FunctionType
from openai import OpenAI
import ollama
from Script.Core import cache_control, game_type, value_handle, get_text
from Script.Design import map_handle, constant, settle_behavior
from Script.UI.Panel import draw_event_text_panel
from Script.Config import normal_config, game_config
from Script.UI.Model import draw

cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """
window_width: int = normal_config.config_normal.text_width
""" 窗体宽度 """
_: FunctionType = get_text._
""" 翻译api """


def handle_event(character_id: int, start: int, now_time: int, end_time: int) -> (draw_event_text_panel.DrawEventTextPanel, str):
    """
    处理状态触发事件
    Keyword arguments:
    character_id -- 角色id
    start -- 是否是状态开始
    start_time -- 事件的开始时间
    end_time -- 事件的结束时间
    Return arguments:
    draw.LineFeedWaitDraw -- 事件绘制文本
    str -- 事件id
    """
    character_data: game_type.Character = cache.character_data[character_id]
    behavior_id = character_data.behavior.behavior_id
    now_event_data = {}
    if (
            behavior_id in game_config.config_event_status_data
            and start in game_config.config_event_status_data[behavior_id]
    ):
        for event_id in game_config.config_event_status_data[behavior_id][start]:
            now_weight = 1
            event_config = game_config.config_event[event_id]
            if len(event_config.premise):
                now_weight = 0
                for premise in event_config.premise:
                    if premise in character_data.premise_data:
                        if not character_data.premise_data[premise]:
                            now_weight = 0
                            break
                        now_weight += character_data.premise_data[premise]
                    else:
                        now_add_weight = constant.handle_premise_data[premise](character_id)
                        character_data.premise_data[premise] = now_add_weight
                        if now_add_weight:
                            now_weight += now_add_weight
                        else:
                            now_weight = 0
                            break
            if now_weight:
                now_event_data.setdefault(now_weight, set())
                now_event_data[now_weight].add(event_id)
    now_event_id = ""
    if now_event_data:
        event_weight = value_handle.get_rand_value_for_value_region(list(now_event_data.keys()))
        now_event_id = random.choice(list(now_event_data[event_weight]))
    line_feed = draw.NormalDraw()
    line_feed.text = "\n"
    line_feed.draw_event = True
    now_target_id = character_data.ai_target
    if now_event_id == "":
        if not character_id:
            now_draw = draw.NormalDraw()
            now_draw.width = window_width
            if start:
                now_draw.text = _("占位符:无可触发事件，待补完")
            else:
                now_draw.text = _("占位符:无可结算事件，待补完")
            now_draw.draw_event = True
            now_draw.draw()
            line_feed.draw()
        if not start:
            character_data.behavior.temporary_status = game_type.TemporaryStatus()
            character_data.behavior.behavior_id = constant.Behavior.SHARE_BLANKLY
            character_data.ai_target = 0
            character_data.behavior.move_target = []
            character_data.behavior.move_src = []
            character_data.behavior.start_time += character_data.behavior.duration
            character_data.behavior.duration = 0
            character_data.state = constant.CharacterStatus.STATUS_ARDER
        return
    now_event_draw = draw_event_text_panel.DrawEventTextPanel(now_event_id, character_id)
    settle_output = settle_behavior.handle_settle_behavior(character_id, end_time, now_event_id, now_target_id)
    start_text = character_data.behavior.start_event_draw_text
    if not start:
        character_data.behavior.temporary_status = game_type.TemporaryStatus()
        character_data.behavior.behavior_id = constant.Behavior.SHARE_BLANKLY
        character_data.behavior.move_target = []
        character_data.behavior.move_src = []
        character_data.behavior.start_time += character_data.behavior.duration
        character_data.behavior.duration = 0
        character_data.behavior.start_event_draw_text = ""
        character_data.state = constant.CharacterStatus.STATUS_ARDER
        character_data.ai_target = 0
    else:
        character_data.behavior.start_event_draw_text = now_event_draw.text
    climax_draw = settlement_pleasant_sensation(character_id)
    player_data: game_type.Character = cache.character_data[0]
    if (not character_id) or (player_data.target_character_id == character_id):
        if now_event_draw.text != "":
            if  not start:
                npc_name = character_data.name
                action_text = game_config.config_status[character_data.state].name
                end_text = now_event_draw.text
                premise_text_list = [game_config.config_premise[premise_id].premise for premise_id in event_config.premise]
                prompt = _(normal_config.config_normal.prompt).format(npc_name=npc_name, action_text=action_text, start_text=start_text, end_text=end_text, premise_text_list=premise_text_list)
                if normal_config.config_normal.ai_mode == 1:
                    try:
                        ollama.list()
                        return_text = send_event_text_to_ollama(prompt)
                        return_text += "\n" + now_event_draw.text
                        now_event_draw.text = return_text
                        now_event_draw.draw()
                    except Exception:
                        now_draw = draw.WaitDraw()
                        now_draw.text = _("无法链接ollama服务，ollama模式已自动关闭")
                        now_draw.width = window_width
                        now_draw.draw_event = True
                        now_draw.draw()
                        normal_config.change_normal_config("ai_mode", 0)
                        now_event_draw.draw()
                elif normal_config.config_normal.ai_mode == 2:
                    return_text = send_event_text_to_api(prompt)
                    if return_text == "":
                        now_draw = draw.WaitDraw()
                        now_draw.text = _("调用外部api失败，api模式已自动关闭")
                        now_draw.width = window_width
                        now_draw.draw_event = True
                        now_draw.draw()
                        normal_config.change_normal_config("ai_mode", 0)
                        now_event_draw.draw()
                    else:
                        return_text += "\n" + now_event_draw.text
                        now_event_draw.text = return_text
                        now_event_draw.draw()
                else:
                    now_event_draw.draw()
            else:
                now_event_draw.draw()
        line_draw = draw.LineDraw("+", 100)
        line_draw.draw_event = True
        line_draw_judge = False
        if settle_output is not None:
            line_draw.draw()
            line_draw_judge = True
            if settle_output[1]:
                name_draw = draw.NormalDraw()
                name_draw.draw_event = True
                character_name = character_data.name
                if character_data.cid:
                    if character_data.nick_name != "":
                        character_name = character_data.nick_name
                name_draw.text = _("{character_name}的状态结算:").format(character_name=character_name)
                name_draw.width = 100
                name_draw.draw()
                line_feed.draw()
            settle_output[0].draw()
            line_feed.draw()
        if climax_draw is not None:
            if not character_id or not character_data.target_character_id:
                if not line_draw_judge:
                    line_draw.draw()
                    line_draw_judge = True
                climax_draw.draw()
                line_feed.draw()


def send_event_text_to_ollama(prompt: str) -> str:
    """
    调用AI补全事件文本
    Keyword arguments:
    prompt -- 提示词
    Return arguments:
    str -- 返回文本
    """
    try:
        response = ollama.chat(
            model=normal_config.config_normal.ollama_mode,
            messages=[
                {'role': 'user', 'content': prompt}
            ]
        )
        pattern = r"<think>.*?</think>"
        return_text = re.sub(pattern, "", response.message.content, flags=re.DOTALL)
        return return_text
    except Exception:
        return ""


def send_event_text_to_api(prompt: str) -> str:
    """
    调用外部api补全事件文本
    Keyword arguments:
    prompt -- 提示词
    Return arguments:
    str -- 返回文本
    """
    try:
        client = OpenAI(api_key=normal_config.config_normal.ai_api_key,base_url=normal_config.config_normal.ai_api_url)
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=normal_config.config_normal.ai_api_model,
        )
        return chat_completion.choices[0].message.content
    except Exception:
        return ""


def settlement_pleasant_sensation(character_id: int) -> draw.NormalDraw():
    """
    结算角色快感
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    draw.NormalDraw -- 高潮结算绘制文本
    """
    character_data: game_type.Character = cache.character_data[character_id]
    behavior_data: game_type.Behavior = character_data.behavior
    climax_data: Dict[int, Set] = {1: set(), 2: set()}
    for organ in character_data.sex_experience:
        organ_config = game_config.config_organ[organ]
        status_id = organ_config.status_id
        if status_id in character_data.status:
            if character_data.status[status_id] >= 1000:
                now_level = 1
                if character_data.status[status_id] >= 10000:
                    now_level = 2
                if status_id == 0:
                    behavior_data.temporary_status.mouth_climax = now_level
                elif status_id == 1:
                    behavior_data.temporary_status.chest_climax = now_level
                elif status_id == 2:
                    behavior_data.temporary_status.vagina_climax = now_level
                elif status_id == 3:
                    behavior_data.temporary_status.clitoris_climax = now_level
                elif status_id == 4:
                    behavior_data.temporary_status.anus_climax = now_level
                elif status_id == 5:
                    behavior_data.temporary_status.penis_climax = now_level
                if now_level == 1:
                    character_data.status[21] *= 0.5
                elif now_level == 2:
                    character_data.status[21] *= 0.8
                character_data.status[21] = max(character_data.status[21], 0)
                character_data.status[status_id] = 0
                climax_data[now_level].add(organ)
                if now_level == 1:
                    character_data.sex_experience[organ] *= 1.01
                else:
                    character_data.sex_experience[organ] *= 1.1
    low_climax_text = ""
    for organ in climax_data[1]:
        organ_config = game_config.config_organ[organ]
        if not low_climax_text:
            low_climax_text += organ_config.name
        else:
            low_climax_text += "+" + organ_config.name
    if low_climax_text:
        low_climax_text += _("绝顶")
    hight_climax_text = ""
    for organ in climax_data[2]:
        organ_config = game_config.config_organ[organ]
        if not hight_climax_text:
            hight_climax_text += organ_config.name
        else:
            hight_climax_text += "+" + organ_config.name
    if hight_climax_text:
        hight_climax_text += _("强绝顶")
    draw_list = []
    if low_climax_text:
        draw_list.append(low_climax_text)
    if hight_climax_text:
        draw_list.append(hight_climax_text)
    if draw_list:
        draw_list.insert(0, character_data.name + ":")
        draw_text = ""
        for i in range(len(draw_list)):
            if i:
                draw_text += " " + draw_list[i]
            else:
                draw_text += draw_list[i]
        now_draw = draw.NormalDraw()
        now_draw.text = draw_text
        now_draw.width = 100
        now_draw.draw_event = True
        return now_draw
    return None
