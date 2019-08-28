from script.Core import EraPrint,TextLoading
from script.Design import AttrText,CmdButtonQueue

def seeCharacterWearClothes(characterId:str):
    '''
    查看角色已穿戴服装列表面板
    Keyword arguments:
    characterId -- 角色id
    '''
    sceneInfo = TextLoading.getTextData(TextLoading.stageWordPath,'79')
    EraPrint.plt(sceneInfo)
    characterInfo = AttrText.getCharacterAbbreviationsInfo(characterId)
    EraPrint.p(characterInfo)

def seeCharacterWearClothesCmd() -> str:
    '''
    用于控制查看角色已装备服装列表面板的命令菜单
    '''
    EraPrint.pline()
    yrn = CmdButtonQueue.optionint(CmdButtonQueue.seecharacterwearclothes,idSize='center',askfor=False)
    return yrn
