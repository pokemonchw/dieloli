import time
from script.Core import GameConfig,TextHandle,FlowHandle,IoInit,CacheContorl,Dictionaries,RichText,TextLoading

last_char = '\n'

#默认输出样式
def_style = IoInit.style_def

#基本输出
def p(string, style='standard'):
    barlist = TextLoading.getTextData(TextLoading.barListId,'barlist')
    styleList = RichText.setRichTextPrint(string, style)
    global last_char
    if len(string) > 0:
        last_char = string[-1:]
    string = RichText.removeRichCache(string)
    for i in range(0,len(string)):
        if styleList[i] in barlist:
            styledata = TextLoading.getTextData(TextLoading.barListId,styleList[i])
            truebar = styledata['truebar']
            nullbar = styledata['nullbar']
            if string[i] == '0':
                pimage(nullbar, 'bar')
            elif string[i] == '1':
                pimage(truebar, 'bar')
            else:
                IoInit.print(string[i], styleList[i])
        else:
            IoInit.print(string[i], styleList[i])

# 输出图片
def pimage(imageName,imagePath=''):
    IoInit.imageprint(imageName, imagePath)
    pass

# 小标题输出
def plt(string):
    textWait = CacheContorl.textWait
    if textWait != 0:
        time.sleep(textWait)
    else:
        pass
    string = str(string)
    string = Dictionaries.handleText(string)
    global last_char
    if len(string) > 0:
        last_char = string[-1:]
    width = GameConfig.text_width
    textWidth = TextHandle.getTextIndex(string)
    lineWidth = int(int(width)/2 - int(textWidth)/2 - 2)
    pl('='*lineWidth + '<littletitle>▢' + string + '▢</littletitle>' + '='*lineWidth)

# 子标题输出
def sontitleprint(string):
    textWait = CacheContorl.textWait
    if textWait != 0:
        time.sleep(textWait)
    else:
        pass
    string = string
    string = Dictionaries.handleText(string)
    global last_char
    if len(string) > 0:
        last_char = string[-1:]
    width = GameConfig.text_width
    textWidth = TextHandle.getTextIndex(string)
    lineWidth = int(int(width)/4)
    lineWidthFix = int(int(width)/4 - int(textWidth))
    pl(':' * lineWidthFix + '<sontitle>' + string + '</sontitle>' + ':' * lineWidth * 3)

#输出一行
def pl(string='', style='standard'):
    textWait = CacheContorl.textWait
    if textWait != 0:
        time.sleep(textWait)
    else:
        pass
    global last_char
    if not last_char == '\n':
        p('\n')
    p(str(string), style)
    if not last_char == '\n':
        p('\n')

#输出分割线
def pline(sample='=', style='standard'):
    textWait = CacheContorl.textWait
    if textWait != 0:
        time.sleep(textWait)
    else:
        pass
    textWidth = GameConfig.text_width
    pl(sample * textWidth,style)

def plittleline(sample = ':',style = 'standard'):
    textWait = CacheContorl.textWait
    if textWait != 0:
        time.sleep(textWait)
    else:
        pass
    textWidth = GameConfig.text_width
    pl(sample * textWidth, style)

# 输出页数线
def printPageLine(sample = ':',string = '',style = 'standard'):
    textWait = CacheContorl.textWait
    if textWait != 0:
        time.sleep(textWait)
    else:
        pass
    textWidth = int(GameConfig.text_width)
    stringWidth = int(TextHandle.getTextIndex(string))
    fixText = sample * int(textWidth / 2 - stringWidth / 2)
    stringText = fixText + string + fixText
    p(stringText,style)

#输出警告
def pwarn(string, style='warning'):
    textWait = CacheContorl.textWait
    if textWait != 0:
        time.sleep(textWait)
    else:
        pass
    """输出警告"""
    pl(string, style)
    print(string)

#输出并等待
def pwait(string, style='standard'):
    textWait = CacheContorl.textWait
    if textWait != 0:
        time.sleep(textWait)
    else:
        pass
    p(string, style)
    FlowHandle.askfor_wait()

#输出一行并等待
def plwait(string='', style='standard'):
    textWait = CacheContorl.textWait
    if textWait != 0:
        time.sleep(textWait)
    else:
        pass
    pl(string, style)
    FlowHandle.askfor_wait()

#逐字输出
def pobo(sleepTime,string, style='standard'):
    textWait = CacheContorl.textWait
    if textWait != 0:
        time.sleep(textWait)
    else:
        pass
    CacheContorl.wframeMouse['wFrameUp'] = 0
    styleList = RichText.setRichTextPrint(string,style)
    styleNameList = GameConfig.getFontDataList()
    for i in range(0,len(styleNameList)):
        styleTextHead = '<' + styleNameList[i] + '>'
        styleTextTail = '</' + styleNameList[i] + '>'
        if styleTextHead in string:
            string = string.replace(styleTextHead,'')
            string = string.replace(styleTextTail, '')
        else:
            pass
    index = len(string)
    for i in range(0,index):
        p(string[i],styleList[i])
        time.sleep(sleepTime)
        if CacheContorl.wframeMouse['wFrameUp'] == 1:
            indexI = i + 1
            CacheContorl.wframeMouse['wFrameUp'] = 2
            for indexI in range(indexI,index):
                p(string[indexI],styleList[indexI])
            if CacheContorl.wframeMouse['wFrameLineState'] == 2:
                CacheContorl.wframeMouse['wFrameLinesUp'] = 2
            break

# 列表输出
def plist(stringList,stringColumn = 1,stringSize = 'left'):
    textWait = CacheContorl.textWait
    if textWait != 0:
        time.sleep(textWait)
    else:
        pass
    textWidth = GameConfig.text_width
    stringIndex = int(textWidth / stringColumn)
    for i in range(0, len(stringList)):
        stringText = stringList[i]
        stringIdIndex = TextHandle.getTextIndex(stringList[i])
        if stringSize == 'left':
            stringTextFix = ' ' * (stringIndex - stringIdIndex)
            stringText = stringText + stringTextFix
        elif stringSize == 'center':
            stringTextFix = ' ' * int((stringIndex - stringIdIndex) / 2)
            stringText = stringTextFix + stringText + stringTextFix
        elif stringSize == 'right':
            stringTextFix = ' ' * (stringIndex - stringIdIndex)
            stringText = stringTextFix + stringText
        if i == 0:
            p(stringText)
        elif i / stringColumn >= 1 and i % stringColumn == 0:
            p('\n')
            p(stringText)
        else:
            p(stringText)

#切换下一屏
def pnextscreen():
    textWait = CacheContorl.textWait
    if textWait != 0:
        time.sleep(textWait)
    else:
        pass
    p('\n' * GameConfig.text_hight)

#多行居中逐字输出
def lcp(sleepTime,string='',style='standard'):
    textWait = CacheContorl.textWait
    if textWait != 0:
        time.sleep(textWait)
    else:
        pass
    CacheContorl.wframeMouse['wFrameLineState'] = 1
    string = str(string)
    stringlist = string.split('\n')
    width = GameConfig.text_width
    styleNameList = GameConfig.getFontDataList()
    stringCenterList = ''
    for i in range(0, len(styleNameList)):
        styleTextHead = '<' + styleNameList[i] + '>'
        styleTextTail = '</' + styleNameList[i] + '>'
        if styleTextHead in string:
            stringCenter = string.replace(styleTextHead, '')
            stringCenter = stringCenter.replace(styleTextTail, '')
            stringCenterList = stringCenter.split('\n')
        else:
            stringCenterList = stringlist
    for i in range(0,len(stringlist)):
        widthI = int(width) / 2
        countIndex = TextHandle.getTextIndex(stringCenterList[i])
        countI = int(countIndex) / 2
        if CacheContorl.wframeMouse['wFrameRePrint'] == 1:
            p('\n')
            p(' ' * int((widthI - countI)))
            p(stringlist[i])
        else:
            p(' ' * int((widthI - countI)))
            pobo(sleepTime, stringlist[i])
            p('\n')
            if CacheContorl.wframeMouse['wFrameLinesUp'] == 1:
                indexIUp = i + 1
                CacheContorl.wframeMouse['wFrameLinesUp'] = 2
                for indexIUp in range(indexIUp, len(stringlist)):
                    pl(TextHandle.align(stringlist[indexIUp], 'center'), style)
                CacheContorl.wframeMouse['wFrameLineState'] = 2
                break
    CacheContorl.wframeMouse['wFrameRePrint'] = 0

#多行回车逐行输出
def lkeyp(string=''):
    textWait = CacheContorl.textWait
    if textWait != 0:
        time.sleep(textWait)
    else:
        pass
    CacheContorl.wframeMouse['wFrameMouseNextLine'] = 1
    string = str(string)
    stringlist = string.split('\n')
    for i in range(0,len(stringlist)):
        plwait(stringlist[i])
    CacheContorl.wframeMouse['wFrameMouseNextLine'] = 0