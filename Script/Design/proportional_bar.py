from Script.Core import game_config, text_handle


def get_proportional_bar(
    value_name: str, max_value: int, value: int, bar_id: str, text_width=0
) -> str:
    """
    通用用于计算比例条的函数
    Keyword arguments:
    value_name -- 比例条名字
    max_value -- 最大数值
    value -- 当前数值
    bar_id -- 用于填充比例条的图形id
    text_width -- 进度条区域宽度 (default 0)
    """
    if text_width == 0:
        text_width = game_config.text_width
    bar_width = (
        text_width
        - text_handle.get_text_index(value_name)
        - 5
        - text_handle.get_text_index(str(max_value))
        - text_handle.get_text_index(str(value))
    )
    proportion = int(int(value) / int(max_value) * bar_width)
    true_bar = "1"
    null_bar = "0"
    proportion_bar = true_bar * proportion
    fix_proportion_bar = null_bar * int(bar_width - proportion)
    proportion_bar = (
        "<" + bar_id + ">" + proportion_bar + fix_proportion_bar + "</" + bar_id + ">"
    )
    proportion_bar = (
        str(value_name)
        + "["
        + proportion_bar
        + "]"
        + "("
        + str(value)
        + "/"
        + str(max_value)
        + ")"
    )
    return proportion_bar


def get_count_bar(value_name: str, max_value: int, value: int, bar_id: str) -> str:
    """
    通用用于计算计数条的函数
    Keyword arguments:
    value_name -- 比例条名字
    max_value -- 最大数值
    value -- 当前数值
    bar_id -- 用于填充比例条的图形id
    """
    true_bar = "1"
    null_bar = "0"
    count_bar = true_bar * int(value)
    fix_count_bar = null_bar * (int(max_value) - int(value))
    count_bar = "<" + bar_id + ">" + count_bar + fix_count_bar + "</" + bar_id + ">"
    count_bar = str(value_name) + "[" + count_bar + "]"
    return count_bar
