from script.Panel import TitleFramePanel
from script.Core import PyCmd,CacheContorl
import os

def titleFrame_func():
    '''
    标题界面绘制流程
    '''
    TitleFramePanel.loadGamePanel()
    CacheContorl.wframeMouse['wFrameRePrint'] = 1
    ans = TitleFramePanel.gameMainPanel()
    PyCmd.clr_cmd()
    if ans == 0:
        CacheContorl.temporaryCharacter = CacheContorl.temporaryCharacterBak.copy()
        CacheContorl.nowFlowId = 'creator_character'
    elif ans == 1:
        CacheContorl.oldFlowId = 'title_frame'
        CacheContorl.nowFlowId = 'load_save'
    elif ans == 2:
        os._exit(0)
