from script.Core import EraPrint,TextLoading
from script.Design import AttrText,CmdButtonQueue

# 查看角色已装备服装列表面板
def seeCharacterWearClothes(characterId):
    sceneInfo = TextLoading.getTextData(TextLoading.stageWordPath,'79')
    EraPrint.plt(sceneInfo)
    characterInfo = AttrText.getCharacterAbbreviationsInfo(characterId)
    EraPrint.p(characterInfo)

# 查看角色服装面板菜单
def seeCharacterWearClothesCmd():
    EraPrint.pline()
    yrn = CmdButtonQueue.optionint(CmdButtonQueue.seecharacterwearclothes,idSize='center',askfor=False)
    return yrn
