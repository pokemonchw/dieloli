import time
from script.Core import GameConfig,TextHandle,FlowHandle,IoInit,CacheContorl,Dictionaries,RichText,TextLoading

last_char = '\n'

#默认输出样式
def_style = IoInit.style_def

def p(string:str, style='standard',richTextJudge=True):
    '''
    游戏基础的文本绘制实现
    Keyword arguments:
    string -- 需要绘制的文本
    style -- 文本的默认样式 (default 'standard')
    richTextJudge -- 启用富文本的开关 (default True)
    '''
    if richTextJudge:
        barlist = TextLoading.getTextData(TextLoading.barConfigPath,'barlist')
        styleList = RichText.setRichTextPrint(string, style)
        global last_char
        if len(string) > 0:
            last_char = string[-1:]
        string = RichText.removeRichCache(string)
        string = r'' + string
        for i in range(0,len(string)):
            if styleList[i] in barlist:
                styledata = TextLoading.getTextData(TextLoading.barConfigPath,styleList[i])
                truebar = styledata['truebar']
                nullbar = styledata['nullbar']
                if string[i] == '0':
                    pimage(nullbar, 'bar')
                elif string[i] == '1':
                    pimage(truebar, 'bar')
                else:
                    IoInit.eraPrint(string[i], styleList[i])
            else:
                IoInit.eraPrint(string[i], styleList[i])
    else:
        IoInit.eraPrint(string,style)

def pimage(imageName:str,imagePath=''):
    '''
    图片绘制在EraPrint中的封装
    Keyword arguments:
    imageName -- 图片id
    imagePath -- 图片所在路径 (default '')
    '''
    IoInit.imageprint(imageName, imagePath)

def plt(string:str):
    '''
    按预订样式"littletitle(小标题)"绘制文本
    示例:
    ====▢小标题▢====
    文本将用=补全至与行同宽
    Keyword arguments:
    string -- 小标题文本
    '''
    textWait = CacheContorl.textWait
    if textWait != 0:
        time.sleep(textWait)
    string = str(string)
    string = Dictionaries.handleText(string)
    global last_char
    if len(string) > 0:
        last_char = string[-1:]
    width = GameConfig.text_width
    textWidth = TextHandle.getTextIndex(string)
    lineWidth = int(int(width)/2 - int(textWidth)/2 - 2)
    pl('='*lineWidth + '<littletitle>▢' + string + '▢</littletitle>' + '='*lineWidth)

def sontitleprint(string:str):
    '''
    按预订样式"sontitle(子标题)"绘制文本
    示例：
    ::::子标题::::
    文本将用=补全至与行同宽
    Keyword arguments:
    string -- 子标题文本
    '''
    textWait = CacheContorl.textWait
    if textWait != 0:
        time.sleep(textWait)
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

def pl(string='', style='standard'):
    '''
    绘制文本并换行
    Keyword arguments:
    string -- 要绘制的文本
    style -- 文本的默认样式 (default 'standard')
    '''
    textWait = CacheContorl.textWait
    if textWait != 0:
        time.sleep(textWait)
    global last_char
    if not last_char == '\n':
        p('\n')
    p(str(string), style)
    if not last_char == '\n':
        p('\n')

def pline(sample='=', style='standard'):
    '''
    绘制一行指定文本
    Keyword arguments:
    string -- 要绘制的文本 (default '=')
    style -- 文本的默认样式 (default 'standard')
    '''
    textWait = CacheContorl.textWait
    if textWait != 0:
        time.sleep(textWait)
    textWidth = GameConfig.text_width
    pl(sample * textWidth,style)

def plittleline():
    '''
    绘制标题线，字符为':'
    '''
    pline(':')

def printPageLine(sample = ':',string = '',style = 'standard'):
    '''
    绘制页数线
    Keyword arguments:
    sample -- 填充线样式 (default ':')
    string -- 页数字符串 (default '')
    style -- 页数线默认样式 (default 'standard')
    '''
    textWait = CacheContorl.textWait
    if textWait != 0:
        time.sleep(textWait)
    textWidth = int(GameConfig.text_width)
    stringWidth = int(TextHandle.getTextIndex(string))
    fixText = sample * int(textWidth / 2 - stringWidth / 2)
    stringText = fixText + string + fixText
    p(stringText,style)

def pwarn(string:str, style='warning'):
    '''
    绘制警告信息(将同时在终端打印)
    Keyword arguments:
    string -- 警告信息文本
    style -- 警告信息的默认样式 (default 'warning')
    '''
    textWait = CacheContorl.textWait
    if textWait != 0:
        time.sleep(textWait)
    pl(string, style)
    print(string)

def pwait(string:str, style='standard'):
    '''
    绘制文本并等待玩家按下回车或鼠标左键
    Keyword arguments:
    string -- 要绘制的文本
    style -- 绘制文本的默认样式 (default 'standard')
    '''
    textWait = CacheContorl.textWait
    if textWait != 0:
        time.sleep(textWait)
    p(string, style)
    FlowHandle.askfor_wait()

def plwait(string='', style='standard'):
    '''
    绘制文本换行并等待玩家按下回车或鼠标左键
    Keyword arguments:
    string -- 要绘制的文本 (default '')
    style -- 绘制文本的默认样式 (default 'standard')
    '''
    textWait = CacheContorl.textWait
    if textWait != 0:
        time.sleep(textWait)
    pl(string, style)
    FlowHandle.askfor_wait()

def pobo(sleepTime:float,string:str, style='standard'):
    '''
    逐字绘制文本
    Keyword arguments:
    sleepTime -- 逐字绘制时，绘制间隔时间
    string -- 需要逐字绘制的文本
    style -- 绘制文本的默认样式 (default 'standard')
    '''
    textWait = CacheContorl.textWait
    if textWait != 0:
        time.sleep(textWait)
    CacheContorl.wframeMouse['wFrameUp'] = 0
    styleList = RichText.setRichTextPrint(string,style)
    styleNameList = GameConfig.getFontDataList()
    for i in range(0,len(styleNameList)):
        styleTextHead = '<' + styleNameList[i] + '>'
        styleTextTail = '</' + styleNameList[i] + '>'
        if styleTextHead in string:
            string = string.replace(styleTextHead,'')
            string = string.replace(styleTextTail, '')
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

def plist(stringList:list,stringColumn=1,stringSize='left'):
    '''
    绘制字符串列表
    Keyword arguments:
    stringList -- 要进行绘制的字符串列表
    stringColum -- 每行的绘制数量(列宽由行宽平分为行数而来) (default 1)
    stringSize -- 每列在列宽中的对齐方式(left/center/right) (default 'left')
    '''
    textWait = CacheContorl.textWait
    textWidth = GameConfig.text_width
    if textWait != 0:
        time.sleep(textWait)
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
            nowTextIndex = TextHandle.getTextIndex(stringText)
            if stringTextFix != '' and nowTextIndex < stringIndex:
                stringText += ' ' * (stringIndex - nowTextIndex)
            elif stringTextFix != '' and nowTextIndex > stringIndex:
                stringText = stringText[-1]
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

def pnextscreen():
    '''
    绘制一整屏空行
    '''
    textWait = CacheContorl.textWait
    if textWait != 0:
        time.sleep(textWait)
    p('\n' * GameConfig.text_hight)

def lcp(sleepTime:float,string='',style='standard'):
    '''
    将多行文本以居中的对齐方式进行逐字绘制
    Keyword arguments:
    sleepTime -- 逐字的间隔时间
    string -- 需要逐字绘制的文本
    style -- 文本的默认样式
    '''
    textWait = CacheContorl.textWait
    if textWait != 0:
        time.sleep(textWait)
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
    '''
    绘制多行文本，并在绘制时，当玩家输入回车，才绘制下一行
    Keyword arguments:
    string -- 要绘制的文本
    '''
    textWait = CacheContorl.textWait
    if textWait != 0:
        time.sleep(textWait)
    CacheContorl.wframeMouse['wFrameMouseNextLine'] = 1
    string = str(string)
    stringlist = string.split('\n')
    for i in range(0,len(stringlist)):
        plwait(stringlist[i])
    CacheContorl.wframeMouse['wFrameMouseNextLine'] = 0
