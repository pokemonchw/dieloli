import time
import core.GameConfig as config
import core.TextHandle as text
import core.flow as flow
import core.pyio as pyio
import core.CacheContorl as cache
import core.Dictionaries as doctionaries

last_char = '\n'

#默认输出样式
def_style = pyio.style_def

#基本输出
def p(string, style='standard'):
    string=str(string)
    string = doctionaries.handleText(string)
    global last_char
    if len(string) > 0:
        last_char = string[-1:]
    pyio.print(string, style)

# 小标题输出
def plt(string,style='standard'):
    string=str(string)
    width = config.text_width
    textWidth = text.getTextIndex(string)
    lineWidth = int(int(width)/2 - int(textWidth)/2 - 2)
    pl('-'*lineWidth + '□' + string + '□' + '-'*lineWidth)

#输出一行
def pl(string='', style='standard'):
    global last_char
    if not last_char == '\n':
        p('\n')
    p(str(string), style)
    if not last_char == '\n':
        p('\n')

#输出分割线
def pline(sample='=', style='standard'):
    textWidth = config.text_width
    pl("-" * textWidth)

#输出警告
def pwarn(string, style='warning'):
    """输出警告"""
    pl(string, style)
    print(string)

#输出并等待
def pwait(string, style='standard'):
    p(string, style)
    flow.askfor_wait()

#输出一行并等待
def plwait(string='', style='standard'):
    pl(string, style)
    flow.askfor_wait()

#逐字输出
def pobo(sleepTime,string, style='standard'):
    cache.wframeMouse['wFrameUp'] = 0
    index = len(string)
    for i in range(0,index):
        p(string[i],style)
        time.sleep(sleepTime)
        if cache.wframeMouse['wFrameUp'] == 1:
            indexI = i + 1
            cache.wframeMouse['wFrameUp'] = 2
            for indexI in range(indexI,index):
                p(string[indexI],style)
            if cache.wframeMouse['wFrameLineState'] == 2:
                cache.wframeMouse['wFrameLinesUp'] = 2
            break

#切换下一屏
def pnextscreen(string = '',style='standard'):
    p('\n' * config.text_hight)

#多行居中逐字输出
def lcp(sleepTime,string='',style='standard'):
    cache.wframeMouse['wFrameLineState'] = 1
    string = str(string)
    stringlist = string.split('\n')
    width = config.text_width
    for i in range(0,len(stringlist)):
        widthI = int(width) / 2
        countIndex = text.getTextIndex(stringlist[i])
        countI = int(countIndex) / 2
        if cache.wframeMouse['wFrameRePrint'] == 1:
            pl(' ' * int((widthI - countI)) + stringlist[i])
        else:
            p(' ' * int((widthI - countI)))
            pobo(sleepTime, stringlist[i])
            p('\n')
            if cache.wframeMouse['wFrameLinesUp'] == 1:
                indexIUp = i + 1
                cache.wframeMouse['wFrameLinesUp'] = 2
                for indexIUp in range(indexIUp, len(stringlist)):
                    pl(text.align(stringlist[indexIUp], 'center'), style)
                cache.wframeMouse['wFrameLineState'] = 2
                break
    cache.wframeMouse['wFrameRePrint'] = 0
