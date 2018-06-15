from core import EraPrint,TextLoading
from design import AttrText

# 查看角色已装备服装列表面板
def seePlayerWearClothes(playerId):
    sceneInfo = TextLoading.getTextData(TextLoading.stageWordId,'79')
    EraPrint.plt(sceneInfo)
    playerInfo = AttrText.getPlayerAbbreviationsInfo(playerId)
    EraPrint.p(playerInfo)
