import core.GameConfig as config

#文本对齐
def align(text,just='left'):
    text = str(text)
    count = len(text)
    countIndex = 0
    for i in range(0,count):
        countIndex = countIndex + get_width(ord(text[i]))
    width = config.text_width
    if just == "right":
        return " " * (width - countIndex - 1) + text
    elif just == "left":
        return text
    elif just == "center":
        widthI = width/2
        countI = countIndex/2
        return " " * int(widthI - countI) + text

def get_width( o ):
    """计算字符宽度"""
    global widths
    if o == 0xe or o == 0xf:
        return 0
    for num, wid in widths:
        if o <= num:
            return wid
    return 1

widths = [
    (126,    1), (159,    0), (687,     1), (710,   0), (711,   1),
    (727,    0), (733,    1), (879,     0), (1154,  1), (1161,  0),
    (4347,   1), (4447,   2), (7467,    1), (7521,  0), (8369,  1),
    (8426,   0), (9000,   1), (9002,    2), (11021, 1), (12350, 2),
    (12351,  1), (12438,  2), (12442,   0), (19893, 2), (19967, 1),
    (55203,  2), (63743,  1), (64106,   2), (65039, 1), (65059, 0),
    (65131,  2), (65279,  1), (65376,   2), (65500, 1), (65510, 2),
    (120831, 1), (262141, 2), (1114109, 1),
]