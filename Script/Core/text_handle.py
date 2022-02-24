from wcwidth import wcswidth
from Script.Config import game_config, normal_config


def align(text: str, just="left", only_fix=False, columns=1, text_width=None) -> str:
    """
    文本对齐处理函数
    Keyword arguments:
    text -- 需要进行对齐处理的文本
    just -- 文本的对齐方式(right/center/left) (default 'left')
    only_fix -- 只返回对齐所需要的补全文本 (default False)
    columns -- 将行宽平分指定列后，再进行对齐补全 (default 1)
    text_width -- 指定行宽，为None时将使用game_config中的配置 (default None)
    Return arguments:
    str -- 对齐后的文本
    """
    count_index = get_text_index(text)
    if text_width is None:
        width = normal_config.config_normal.text_width
        width = int(width / columns)
    else:
        width = int(text_width)
    if just == "right":
        if only_fix:
            return " " * (width - count_index)
        return " " * (width - count_index) + text
    if just == "left":
        if only_fix:
            return " " * (width - count_index)
        return text + " " * (width - count_index)
    if just == "center":
        left_text = ""
        right_text = ""
        if width < count_index:
            now_text = ""
            if width > 0:
                for i in text:
                    if get_text_index(now_text) + get_text_index(i) < width:
                        now_text += i
                    else:
                        break
                now_text = now_text[:-2] + "~"
        elif width != count_index:
            for i in range(width - count_index):
                if not i % 2:
                    left_text += " "
                else:
                    right_text += " "
        if only_fix:
            return left_text
        return f"{left_text}{text}{right_text}"


def get_text_index(text: str) -> int:
    """
    计算文本最终显示的真实长度
    Keyword arguments:
    text -- 要进行长度计算的文本
    """
    text_index = 0
    style_width = 0
    style_name_list = list(game_config.config_font_data.keys())
    for i in range(0, len(style_name_list)):
        style_text_head = "<" + style_name_list[i] + ">"
        style_text_tail = "</" + style_name_list[i] + ">"
        if style_text_head in text:
            text = text.replace(style_text_head, "")
            text = text.replace(style_text_tail, "")
    for _unused, now_text in enumerate(text):
        text_index += wcswidth(now_text)
    now_width = text_index + style_width
    now_width = max(now_width, 0)
    return now_width


def full_to_half_text(ustring: str) -> str:
    """
    将全角字符串转换为半角
    Keyword arguments:
    ustring -- 要转换的全角字符串
    """
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 12288:
            inside_code = 32
        elif 65281 <= inside_code <= 65374:
            inside_code -= 65248
        aaa = chr(inside_code)
        rstring += aaa
    return rstring


def id_index(now_id: int) -> str:
    """
    生成命令id文本
    Keyword arguments:
    now_id -- 命令id
    Return arguments:
    str -- id文本
    """
    return f"[{str(now_id).zfill(3)}]"


def number_to_symbol_string(value: int) -> str:
    """
    数字转换为带正负数符号的数字字符串
    Keyword arguments:
    value -- 要转换的数字
    Return arguments:
    str -- 转换后的字符串
    """
    symbol = ""
    if value >= 0:
        symbol = "+"
    return f"{symbol}{value}"
