import script.Panel.MainFramePanel as mainframepanel
import core.CacheContorl as cache
import script.AttrHandle as attrhandle
import core.game as game
import core.PyCmd as pycmd
import script.flow.SeePlayerAttr as seeplayerattr
import core.ValueHandle as valuehandle

# 游戏主页
def mainFrame_func():
    inputS = []
    flowReturn = mainframepanel.mainFramePanel()
    inputS = valuehandle.listAppendToList(flowReturn,inputS)
    askForMainFrame(inputS)

# 主页控制流程
def askForMainFrame(ansList):
    playerId = cache.playObject['objectId']
    playerData = attrhandle.getAttrData(playerId)
    playerName = playerData['Name']
    ans = game.askfor_All(ansList)
    if ans == playerName:
        pycmd.clr_cmd()
        mainFrameSeeAttrPanel()
    else:
        pass

# 游戏主页查看属性流程
def mainFrameSeeAttrPanel():
    seeplayerattr.seeAttrInEveryTime_func()
    pass