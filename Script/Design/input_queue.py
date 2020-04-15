from Script.Core import game_init, era_print, text_loading


def wait_input(int_a: int, int_b: int) -> str:
    """
    等待玩家输入ab之间的一个数
    Keyword arguments:
    int_a -- 输入边界A
    int_b -- 输入边界B
    """
    while True:
        ans = game_init.askfor_str()
        if ans.isdecimal():
            ans = int(ans)
            if int_a <= ans <= int_b:
                break
        era_print.line_feed_print(ans)
        era_print.line_feed_print(
            text_loading.get_text_data(
                text_loading.ERROR_PATH, "input_null_error"
            )
            + "\n"
        )
    return ans
