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


#输出一行
def pl(string='', style='standard'):
    global last_char
    if not last_char == '\n':
        p('\n')
    p(str(string), style)
    if not last_char == '\n':
        p('\n')

#输出分割线
def pline(sample='＝', style='standard'):
    fontName = config.font
    fontSize = config.font_size
    width = text.getWinFrameWidth(sample,fontName,fontSize)
    pl(sample * width, style)

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
            break

#输出标题
def pti(string,style='title'):
    fontSize = config.title_fontsize
    fontName = config.font
    width = int(text.getWinFrameWidth(string,fontName,fontSize))
    p(text.align(string,width,'center'), style)

#切换下一屏
def pnextscreen(string = '',style='standard'):
    winHight = text.getWinFrameHight(config.font,config.font_size)
    p('\n' * winHight * 2)