from Script.Core import cache_contorl, game_config
from Script.Panel import use_item_panel


def scene_see_character_item(character_id: int):
    """
    在场景中查看角色道具列表的流程
    Keyword arguments:
    character_id -- 角色Id
    """
    while 1:
        use_item_panel.see_character_item_panel(character_id)
        cache_contorl.now_flow_id = "main"
        break


def open_character_bag():
    """
    打开主角背包查看道具列表流程
    """
    scene_see_character_item(0)


def get_character_item_page_max(character_id: str):
    """
    计算角色道具列表页数
    Keyword arguments:
    character_id -- 角色Id
    """
    item_max = len(
        cache_contorl.character_data["character"][character_id].item
    )
    page_index = game_config.see_character_item_max
    if item_max - page_index < 0:
        return 0
    elif item_max % page_index == 0:
        return item_max / page_index - 1
    else:
        return int(item_max / page_index)
