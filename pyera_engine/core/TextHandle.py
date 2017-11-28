import core.winframe as winframe
import core.GameConfig as config
import tkinter.font as tkFont

#获取窗体宽度(字数)
def getWinFrameWidth(textS,fontName,fontSizePt):
    indexText = len(textS)
    textLong = 0
    for i in range(0,indexText):
        textLong = textLong + get_width(ord(textS[i]))
    frameWidth = winframe.root.winfo_width()
    textWidth = getFontWidth(textS,fontName,fontSizePt)
    fontSizePx = int(textWidth/textLong)
    width = int(int(frameWidth) / int(fontSizePx))
    return width

#获取字体宽度
def getFontWidth(text,fontName,fontSize):
    fontDpi = config.font_dpi
    font = tkFont.Font(name=fontName, size=int(fontSize))
    textwidth = font.measure(text)
    width = int(textwidth) * int(fontDpi) / 72
    return width

#获取窗体高度(字数)
def getWinFrameHight(fontName,fontSizePt):
    frameHight = winframe.root.winfo_height()
    fontHight = getFontHight("A", fontName, fontSizePt)
    hight = int(int(frameHight)/int(fontHight))
    return hight

#获取字体高度
def getFontHight(text,fontName,fontSize):
    fontDpi = config.font_dpi
    font = tkFont.Font(name=fontName, size=int(fontSize))
    texthight = font.metrics("linespace")
    hight = int(texthight) * int(fontDpi) / 72
    return hight

#文本对齐
def align(text, width, just='left'):
    text = str(text)
    count = len(text)
    if just == "right":
        return " " * (width - count) + text
    elif just == "left":
        return text + " " * (width - count)
    elif just == "center":
        widthI = int(int(width)/2)
        countI = int(int(count)/2)
        return " " * (widthI - countI + 2) + text

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