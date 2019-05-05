from script.Panel import GameHelpPanel
from script.Core import FlowHandle,CacheContorl

# 查看帮助信息流程
def gameHelp_func():
    inputS = GameHelpPanel.gameHelpPanel()
    yrn = FlowHandle.askfor_All(inputS)
    if yrn == '0':
        CacheContorl.nowMap = 'main'
