import core.EraPrint as eprint
import core.flow as flow
import core.TextLoading as textload
import time
import core.TextHandle as text
import core.GameConfig as config
import core.PyCmd as pycmd
import script.Ans as ans

# 载入游戏面板
def loadGamePanel():
    eprint.pnextscreen()
    flow.initCache()
    eprint.pobo(1 / 3, textload.getTextData(textload.messageId, '1'))
    eprint.p('\n')
    pass

# 游戏主面板
def gameMainPanel():
    import script.mainflow as mainflow
    eprint.pline()
    eprint.pl(text.align(config.game_name, 'center'))
    eprint.pl(text.align(config.author, 'right'))
    eprint.pl(text.align(config.verson, 'right'))
    eprint.pl(text.align(config.verson_time, 'right'))
    eprint.p('\n')
    eprint.pline()
    eprint.lcp(1 / 3, textload.getTextData(textload.messageId, '2'))
    time.sleep(1)
    eprint.p('\n')
    eprint.pline()
    time.sleep(1)
    pycmd.focusCmd()
    menuInt = ans.optionint(ans.logomenu)
    if menuInt == 0:
        mainflow.newgame_func()
    elif menuInt == 1:
        mainflow.loadgame_func()
    elif menuInt == 2:
        mainflow.quitgame_func()
    pass