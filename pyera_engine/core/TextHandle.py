from idna import unichr

import core.GameConfig as config
import core.TextLoading as textload
import core.RichText as richtext

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

#文本对齐
def align(text,just='left'):
    text = str(text)
    countIndex = getTextIndex(text)
    width = config.text_width
    if just == "right":
        return " " * (width - countIndex) + text
    elif just == "left":
        return text
    elif just == "center":
        widthI = width/2
        countI = countIndex/2
        return " " * int(widthI - countI) + text

#文本长度计算
def getTextIndex(text):
    textStyleList = richtext.setRichTextPrint(text, 'standard')
    textIndex = 0
    stylewidth = 0
    barlist = textload.getTextData(textload.barListId,'barlist')
    styleNameList = config.getFontData('styleList') + textload.getTextData(textload.barListId,'barlist')
    for i in range(0, len(styleNameList)):
        styleTextHead = '<' + styleNameList[i] + '>'
        styleTextTail = '</' + styleNameList[i] + '>'
        if styleTextHead in text:
            if styleNameList[i] in textload.getTextData(textload.barListId,'barlist'):
                text = text.replace(styleTextHead, '')
                text = text.replace(styleTextTail, '')
            else:
                text = text.replace(styleTextHead, '')
                text = text.replace(styleTextTail, '')
        else:
            pass
    count = len(text)
    for i in range(0,count):
        if textStyleList[i] in barlist:
            textwidth = textload.getTextData(textload.barListId,textStyleList[i])['width']
            textIndex = textIndex + int(textwidth)
        else:
            textIndex = textIndex + get_width(ord(text[i]))
    return textIndex + stylewidth

# 计算字符宽度
def get_width( o ):
    global widths
    if o == 0xe or o == 0xf:
        return 0
    for num, wid in widths:
        if o <= num:
            return wid
    return 1

# 全角字符转半角
def fullToHalfText(ustring):
    rstring = ""
    for uchar in ustring:
        inside_code=ord(uchar)
        if inside_code == 12288:
            inside_code = 32
        elif (inside_code >= 65281 and inside_code <= 65374):
            inside_code -= 65248
        rstring += unichr(inside_code)
    return rstring