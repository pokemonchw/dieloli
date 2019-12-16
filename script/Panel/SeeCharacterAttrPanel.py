from script.Core import CacheContorl,TextLoading,EraPrint,PyCmd,GameConfig
from script.Design import AttrPrint,AttrHandle,AttrText,CmdButtonQueue
from script.Panel import ChangeClothesPanel,UseItemPanel,WearItemPanel

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

def seeCharacterStatusHeadPanel(characterId:str) -> str:
    '''
    查看角色状态面板头部面板
    Keyword arguments:
    characterId -- 角色Id
    '''
    EraPrint.plt(TextLoading.getTextData(TextLoading.stageWordPath, '135'))
    EraPrint.pl(AttrText.getSeeAttrPanelHeadCharacterInfo(characterId))
    seeCharacterStatusPanel(characterId)

def seeCharacterStatusPanel(characterId:str):
    '''
    查看角色状态面板
    Keyword arguments:
    characterId -- 角色Id
    '''
    statusTextData = TextLoading.getTextData(TextLoading.stageWordPath, '134')
    characterData = CacheContorl.characterData['character'][characterId]
    statusData = characterData['Status']
    for stateType in statusData:
        EraPrint.sontitleprint(statusTextData[stateType])
        nowStatusData = statusData[stateType].copy()
        if stateType == 'SexFeel':
            if characterData['Sex'] == 'Man':
                del nowStatusData['VaginaDelight']
                del nowStatusData['ClitorisDelight']
                del nowStatusData['VaginaLubrication']
            elif characterData['Sex'] == 'Woman':
                del nowStatusData['PenisDelight']
            elif characterData['Sex'] == 'Asexual':
                del nowStatusData['VaginaDelight']
                del nowStatusData['ClitorisDelight']
                del nowStatusData['VaginaLubrication']
                del nowStatusData['PenisDelight']
        nowStatusTextList = [statusTextData[state] + ':' + str(nowStatusData[state]) for state in nowStatusData]
        EraPrint.plist(nowStatusTextList,4,'center')
    EraPrint.pl()

def seeCharacterHPAndMPInSence(characterId:str):
    '''
    在场景中显示角色的HP和MP
    Keyword arguments:
    characterId -- 角色Id
    '''
    if characterId == '0':
        AttrPrint.printHpAndMpBar(characterId)
    else:
        characterIdText = TextLoading.getTextData(TextLoading.stageWordPath, '0') + '0' + ':' + CacheContorl.characterData['character']['0']['Name']
        targetIdText = TextLoading.getTextData(TextLoading.stageWordPath, '0') + characterId + ':' + CacheContorl.characterData['character'][characterId]['Name']
        EraPrint.plist([characterIdText,targetIdText],2,'center')
        EraPrint.pl()
        playerBar = AttrPrint.getHpOrMpBar('0','HitPoint',GameConfig.text_width / 2 - 4)
        targetBar = AttrPrint.getHpOrMpBar(characterId,'HitPoint',GameConfig.text_width / 2 - 4)
        EraPrint.plist([playerBar,targetBar],2,'center')
        EraPrint.pl()
        playerBar = AttrPrint.getHpOrMpBar('0','ManaPoint',GameConfig.text_width / 2 - 4)
        targetBar = AttrPrint.getHpOrMpBar(characterId,'ManaPoint',GameConfig.text_width / 2 - 4)
        EraPrint.plist([playerBar,targetBar],2,'center')
        EraPrint.pl()

def seeCharacterEquipmentPanel(characterId:str):
    '''
    查看角色装备面板
    Keyword arguments:
    characterId -- 角色Id
    '''
    EraPrint.plt(TextLoading.getTextData(TextLoading.stageWordPath, '37'))
    EraPrint.p(AttrText.getSeeAttrPanelHeadCharacterInfo(characterId))
    ChangeClothesPanel.seeCharacterWearClothes(characterId,False)

def seeCharacterItemPanel(characterId:str):
    '''
    查看角色道具面板
    Keyword arguments:
    characterId -- 角色Id
    '''
    EraPrint.plt(TextLoading.getTextData(TextLoading.stageWordPath, '38'))
    EraPrint.p(AttrText.getSeeAttrPanelHeadCharacterInfo(characterId))
    UseItemPanel.seeCharacterItemPanel(characterId)

def seeCharacterWearItemPanel(characterId:str):
    '''
    查看角色穿戴道具面板
    Keyword arguments:
    characterId -- 角色Id
    '''
    EraPrint.plt(TextLoading.getTextData(TextLoading.stageWordPath, '40'))
    EraPrint.pl(AttrText.getSeeAttrPanelHeadCharacterInfo(characterId))
    WearItemPanel.seeCharacterWearItemPanel(characterId,False)

def askForSeeAttr() -> list:
    '''
    查看角色属性时输入处理面板
    '''
    EraPrint.pline()
    askData = TextLoading.getTextData(TextLoading.cmdPath,CmdButtonQueue.seeattrpanelmenu).copy()
    nowPanelId = CacheContorl.panelState['AttrShowHandlePanel']
    nullCmd = askData[nowPanelId]
    askList = list(askData.values())
    CmdButtonQueue.optionstr(None,5,'center',False,False,askList,nowPanelId,list(askData.keys()))
    del askData[nowPanelId]
    return list(askData.keys())

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
    "Status":seeCharacterStatusHeadPanel,
    "Item":seeCharacterItemPanel,
    "WearItem":seeCharacterWearItemPanel,
    "SexExperience":"",
    "Knowledge":"",
    "Language":"",
    "Features":"",
    "SocialContact":""
}

