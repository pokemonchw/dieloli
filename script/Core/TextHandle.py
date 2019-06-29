from script.Core import GameConfig,TextLoading,RichText
from wcwidth import wcswidth

#文本对齐
def align(text,just='left',onlyFix = False,columns = 1,textWidth = None):
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

# 文本长度计算
def getTextIndex(text):
    textStyleList = RichText.setRichTextPrint(text, 'standard')
    textIndex = 0
    stylewidth = 0
    barlist = TextLoading.getTextData(TextLoading.barConfigPath,'barlist')
    styleNameList = GameConfig.getFontDataList() + TextLoading.getTextData(TextLoading.barConfigPath,'barlist')
    for i in range(0, len(styleNameList)):
        styleTextHead = '<' + styleNameList[i] + '>'
        styleTextTail = '</' + styleNameList[i] + '>'
        if styleTextHead in text:
            if styleNameList[i] in TextLoading.getTextData(TextLoading.barConfigPath,'barlist'):
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

# 全角字符转半角
def fullToHalfText(ustring):
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
