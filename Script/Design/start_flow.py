# -*- coding: UTF-8 -*-
from Script.Core import py_cmd, cache_contorl, flow_handle
from Script.Flow import (
    creator_character,
    save_handle_frame,
    title_frame,
    main,
    see_character_attr,
    in_scene,
    see_character_list,
    change_clothes,
    see_map,
    wear_item,
    use_item,
)

flow_data = {
    "title_frame": title_frame.title_frame_func,
    "creator_character": creator_character.input_name_func,
    "load_save": save_handle_frame.load_save_func,
    "establish_save": save_handle_frame.establish_save_func,
    "main": main.main_frame_func,
    "see_character_attr": see_character_attr.see_attr_on_every_time_func,
    "in_scene": in_scene.get_in_scene_func,
    "see_character_list": see_character_list.see_character_list_func,
    "change_clothes": change_clothes.change_character_clothes,
    "see_map": see_map.see_map_flow,
    "acknowledgment_attribute": see_character_attr.acknowledgment_attribute_func,
    "open_bag": use_item.open_character_bag,
    "wear_item": wear_item.wear_character_item,
}


def start_frame():
    """
    游戏主流程
    """
    flow_handle.init_cache()
    while True:
        now_flow_id = cache_contorl.now_flow_id
        py_cmd.clr_cmd()
        flow_data[now_flow_id]()
