import time
from script.Core import EraPrint,TextLoading,TextHandle,GameConfig,PyCmd
from script.Design import CmdButtonQueue

def loadGamePanel():
    '''
    载入游戏动画绘制
    '''
    EraPrint.pnextscreen()
    EraPrint.pnextscreen()
    EraPrint.pobo(1 / 3, TextLoading.getTextData(TextLoading.messagePath, '1'))
    EraPrint.p('\n')

def gameMainPanel() -> int:
    '''
    游戏标题界面主面板
    '''
    EraPrint.pline()
    EraPrint.pl(TextHandle.align(GameConfig.game_name, 'center'))
    EraPrint.pl(TextHandle.align(GameConfig.author, 'right'))
    EraPrint.pl(TextHandle.align(GameConfig.verson, 'right'))
    EraPrint.pl(TextHandle.align(GameConfig.verson_time, 'right'))
    EraPrint.p('\n')
    EraPrint.pline()
    EraPrint.lcp(1 / 3, TextLoading.getTextData(TextLoading.messagePath, '2'))
    time.sleep(1)
    EraPrint.p('\n')
    EraPrint.pline()
    time.sleep(1)
    PyCmd.focusCmd()
    menuInt = CmdButtonQueue.optionint(CmdButtonQueue.logomenu)
    return menuInt
