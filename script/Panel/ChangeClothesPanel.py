from script.Core import EraPrint,TextLoading
from script.Design import AttrText,CmdButtonQueue

# 查看角色已装备服装列表面板
def seePlayerWearClothes(playerId):
    sceneInfo = TextLoading.getTextData(TextLoading.stageWordId,'79')
    EraPrint.plt(sceneInfo)
    playerInfo = AttrText.getPlayerAbbreviationsInfo(playerId)
    EraPrint.p(playerInfo)

# 查看角色服装面板菜单
def seePlayerWearClothesCmd():
    EraPrint.pline()
    yrn = CmdButtonQueue.optionint(CmdButtonQueue.seeplayerwearclothes,idSize='center',askfor=False)
    return yrn
