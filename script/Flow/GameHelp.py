from script.Panel import GameHelpPanel
from script.Core import FlowHandle,CacheContorl

def gameHelp_func():
    '''
    查看帮助信息流程
    '''
    inputS = GameHelpPanel.gameHelpPanel()
    yrn = FlowHandle.askfor_All(inputS)
    if yrn == '0':
        CacheContorl.nowFlowId = 'main'
