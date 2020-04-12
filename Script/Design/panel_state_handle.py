from Script.Core import cache_contorl


def panel_state_change(panel_id: str):
    """
    改变面板状态，若该面板当前状态为0，则更改为1，或者反过来
    Keyword arguments:
    panel_id -- 面板id
    """
    cache_panel_state = cache_contorl.panel_state[panel_id]
    if cache_panel_state == "0":
        cache_contorl.panel_state[panel_id] = "1"
    else:
        cache_contorl.panel_state[panel_id] = "0"
