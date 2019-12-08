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
    characterSpecies = characterData['Species']
    characterSpecies = TextLoading.getTextData(TextLoading.stageWordPath, '15') + characterSpecies
    characterAge = characterData['Age']
    characterAge = TextLoading.getTextData(TextLoading.stageWordPath, '3') + str(characterAge)
    EraPrint.plist([characterSpecies,characterAge],2,'center')
    EraPrint.pline('.')
    characterIntimate = characterData['Intimate']
    characterIntimate = TextLoading.getTextData(TextLoading.stageWordPath, '16') + characterIntimate
    characterGraces = characterData['Graces']
    characterGraces = TextLoading.getTextData(TextLoading.stageWordPath, '17') + characterGraces
    EraPrint.plist([characterIntimate,characterGraces],2,'center')
    EraPrint.pline('.')
    characterHeight = characterData['Height']['NowHeight']
    characterWeight = characterData['Weight']
    characterHeightText = str(round(characterHeight,2))
    characterWeightText = str(round(characterWeight,2))
    characterHeightInfo = TextLoading.getTextData(TextLoading.stageWordPath,'80') + characterHeightText
    characterWeightInfo = TextLoading.getTextData(TextLoading.stageWordPath,'81') + characterWeightText
    EraPrint.plist([characterHeightInfo,characterWeightInfo],2,'center')
    EraPrint.pline('.')
    characterMeasurements = characterData['Measurements']
    characterBust = str(round(characterMeasurements['Bust'],2))
    characterWaist = str(round(characterMeasurements['Waist'],2))
    characterHip = str(round(characterMeasurements['Hip'],2))
    characterBustInfo = TextLoading.getTextData(TextLoading.stageWordPath,'82') + characterBust
    characterWaistInfo = TextLoading.getTextData(TextLoading.stageWordPath,'83') + characterWaist
    characterHipInfo = TextLoading.getTextData(TextLoading.stageWordPath,'84') + characterHip
    EraPrint.plist([characterBustInfo,characterWaistInfo,characterHipInfo],3,'center')
    EraPrint.pline('.')

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
    CmdButtonQueue.optionstr(None,3,'center',False,False,askList,nullCmd)
    del askData[nowPanelId]
    return list(askData.values())

def askForSeeAttrCmd() -> list:
    '''
    查看属性页显示控制
    '''
    EraPrint.pline('~')
    yrn = CmdButtonQueue.optionint(CmdButtonQueue.seeattronrverytime, 3, cmdSize='center', askfor=False)
    return yrn

def inputAttrOverPanel():
    '''
    创建角色完成时确认角色属性输入处理面板
    '''
    yrn = CmdButtonQueue.optionint(CmdButtonQueue.acknowledgmentAttribute, askfor=False)
    return yrn

panelData = {
    "MainAttr":seeCharacterMainAttrPanel,
    "Equipment":seeCharacterEquipmentPanel,
    "Status":"",
    "Item":"",
    "SexExperience":"",
    "Knowledge":"",
    "Language":"",
    "Features":"",
    "SocialContact":""
}

