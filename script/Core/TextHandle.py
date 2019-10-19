from script.Core import GameConfig,TextLoading,RichText
from wcwidth import wcswidth

def align(text:str,just='left',onlyFix = False,columns = 1,textWidth = None) -> str:
    '''
    文本对齐处理函数
    Keyword arguments:
    text -- 需要进行对齐处理的文本
    just -- 文本的对齐方式(right/center/left) (default 'left')
    onlyFix -- 只返回对齐所需要的补全文本 (default False)
    columns -- 将行宽平分指定列后，再进行对齐补全 (default 1)
    textWidth -- 指定行宽，为None时将使用GameConfig中的配置 (default None)
    '''
    text = str(text)
    countIndex = getTextIndex(text)
    if textWidth == None:
        width = GameConfig.text_width
        width = int(width / columns)
    else:
        width = int(textWidth)
    if just == "right":
        if onlyFix == True:
            return " " * (width - countIndex)
        else:
            return " " * (width - countIndex) + text
    elif just == "left":
        if onlyFix == True:
            return " " * (width - countIndex)
        else:
            return text + " " * (width - countIndex)
    elif just == "center":
        widthI = width/2
        countI = countIndex/2
        if onlyFix == True:
            return " " * int(widthI - countI)
        else:
            return " " * int(widthI - countI) + text + " " * int(widthI - countI - 2)

def getTextIndex(text:str) -> int:
    '''
    计算文本最终显示的真实长度
    Keyword arguments:
    text -- 要进行长度计算的文本
    '''
    textStyleList = RichText.setRichTextPrint(text, 'standard')
    textIndex = 0
    stylewidth = 0
    barlist = list(TextLoading.getGameData(TextLoading.barConfigPath).keys())
    styleNameList = GameConfig.getFontDataList() + barlist
    for i in range(0, len(styleNameList)):
        styleTextHead = '<' + styleNameList[i] + '>'
        styleTextTail = '</' + styleNameList[i] + '>'
        if styleTextHead in text:
            if styleNameList[i] in barlist:
                text = text.replace(styleTextHead, '')
                text = text.replace(styleTextTail, '')
            else:
                text = text.replace(styleTextHead, '')
                text = text.replace(styleTextTail, '')
    for i in range(len(text)):
        if textStyleList[i] in barlist:
            textwidth = TextLoading.getTextData(TextLoading.barConfigPath,textStyleList[i])['width']
            textIndex = textIndex + int(textwidth)
        else:
            textIndex += wcswidth(text[i])
    return textIndex + stylewidth

def fullToHalfText(ustring:str) -> str:
    '''
    将全角字符串转换为半角
    Keyword arguments:
    ustring -- 要转换的全角字符串
    '''
    rstring = ""
    for uchar in ustring:
        inside_code=ord(uchar)
        if inside_code == 12288:
            inside_code = 32
        elif (inside_code >= 65281 and inside_code <= 65374):
            inside_code -= 65248
        aaa = chr(inside_code)
        rstring += aaa
    return rstring
