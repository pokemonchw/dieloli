from script.Core import CacheContorl,TextLoading,EraPrint,PyCmd,GameConfig
from script.Design import AttrPrint,AttrHandle,AttrText,CmdButtonQueue
from script.Panel import ChangeClothesPanel

def seeCharacterMainAttrPanel(characterId:str):
    '''
    查看角色主属性面板
    Keyword arguments:
    characterId -- 角色Id
    '''
    title1 = TextLoading.getTextData(TextLoading.stageWordPath, '1')
    EraPrint.plt(title1)
    characterIdText = TextLoading.getTextData(TextLoading.stageWordPath, '0') + characterId
    characterData = CacheContorl.characterData['character'][characterId]
    name = characterData['Name']
    nickName = characterData['NickName']
    characterName = TextLoading.getTextData(TextLoading.stageWordPath,'13') + name
    characterNickName = TextLoading.getTextData(TextLoading.stageWordPath,'12') + nickName
    sex = characterData['Sex']
    sexText = TextLoading.getTextData(TextLoading.stageWordPath, '2') + AttrText.getSexText(sex)
    nameText = characterIdText + ' ' + characterName + ' ' + characterNickName + ' ' + sexText
    hpBar = AttrPrint.getHpOrMpBar(characterId,'HitPoint',GameConfig.text_width / 2 - 4)
    EraPrint.plist([nameText,hpBar],2,'center')
    EraPrint.pl()
    stateText = AttrText.getStateText(characterId)
    mpBar = AttrPrint.getHpOrMpBar(characterId,'ManaPoint',GameConfig.text_width / 2 - 4)
    EraPrint.plist([stateText,mpBar],2,'center')
    EraPrint.pl()
    EraPrint.plittleline()
    statureText = AttrText.getStatureText(characterId)
    EraPrint.pl(statureText)
    EraPrint.plittleline()

def seeCharacterEquipmentPanel(characterId:str) -> str:
    '''
    查看角色装备面板
    Keyword arguments:
    characterId -- 角色Id
    '''
    EraPrint.plittleline()
    EraPrint.p(TextLoading.getTextData(TextLoading.stageWordPath, '37'))
    PyCmd.pcmd(panelStateOffText,'CharacterEquipmentPanel')
    characterData = AttrHandle.getAttrData(characterId)
    ChangeClothesPanel.seeCharacterWearClothes(characterId,False)

def askForSeeAttr() -> list:
    '''
    查看角色属性时输入处理面板
    '''
    EraPrint.pline()
    askData = TextLoading.getTextData(TextLoading.cmdPath,CmdButtonQueue.seeattrpanelmenu).copy()
    nowPanelId = CacheContorl.panelState['AttrShowHandlePanel']
    nullCmd = askData[nowPanelId]
    askList = list(askData.values())
    CmdButtonQueue.optionstr(None,4,'center',False,False,askList,nullCmd)
    del askData[nowPanelId]
    return list(askData.values())

def inputAttrOverPanel():
    '''
    创建角色完成时确认角色属性输入处理面板
    '''
    yrn = CmdButtonQueue.optionint(CmdButtonQueue.acknowledgmentAttribute, askfor=False)
    return yrn

panelData = {
    "MainAttr":seeCharacterMainAttrPanel,
    "Equipment":seeCharacterEquipmentPanel,
    "Item":"",
    "SexExperience":"",
    "Knowledge":"",
    "Language":"",
    "Features":""
}

