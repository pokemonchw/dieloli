from Script.Core import cache_contorl, game_init, py_cmd
from Script.Design import clothing
from Script.Panel import main_frame_panel

main_frame_goto_data = {
    "0": "in_scene",
    "1": "see_character_list",
    "2": "change_clothes",
    "3": "wear_item",
    "4": "open_bag",
    "5": "establish_save",
    "6": "load_save",
}


def main_frame_func():
    """
    游戏主页控制流程
    """
    input_s = []
    flow_return = main_frame_panel.main_frame_panel()
    input_s = input_s + flow_return
    character_id = cache_contorl.character_data["character_id"]
    character_data = cache_contorl.character_data["character"][character_id]
    character_name = character_data.name
    ans = game_init.askfor_all(input_s)
    py_cmd.clr_cmd()
    cache_contorl.old_flow_id = "main"
    if ans == character_name:
        cache_contorl.now_flow_id = "see_character_attr"
    else:
        if ans == "0":
            clothing.init_character_clothing_put_on()
        cache_contorl.now_flow_id = main_frame_goto_data[ans]
