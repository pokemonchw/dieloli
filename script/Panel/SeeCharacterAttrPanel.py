from script.Core import CacheContorl,TextLoading,EraPrint,PyCmd
from script.Design import AttrPrint,AttrHandle,AttrText,CmdButtonQueue

panelStateTextData = TextLoading.getTextData(TextLoading.cmdPath,'cmdSwitch')
panelStateOnText = panelStateTextData[1]
panelStateOffText = panelStateTextData[0]

def initShowAttrPanelList():
    '''
    初始化查看属性面板状态
    '''
    CacheContorl.panelState['CharacterMainAttrPanel'] = '1'
    CacheContorl.panelState['CharacterEquipmentPanel'] = '1'
    CacheContorl.panelState['CharacterItemPanel'] = '1'
    CacheContorl.panelState['CharacterExperiencePanel'] = '1'
    CacheContorl.panelState['CharacterLevelPanel'] = '1'
    CacheContorl.panelState['CharacterFeaturesPanel'] = '1'
    CacheContorl.panelState['CharacterEngravingPanel'] = '1'

def seeCharacterMainAttrPanel(characterId:str) -> str:
    '''
    查看角色主属性面板
    Keyword arguments:
    characterId -- 角色Id
    '''
    title1 = TextLoading.getTextData(TextLoading.stageWordPath, '1')
    EraPrint.plt(title1)
    characteridText = TextLoading.getTextData(TextLoading.stageWordPath, '0') + characterId
    EraPrint.p(characteridText)
    panelState = CacheContorl.panelState['CharacterMainAttrPanel']
    if panelState == "0":
        PyCmd.pcmd(panelStateOffText,'CharacterMainAttrPanel',None)
        EraPrint.p('\n')
        attrListString = []
        characterData = AttrHandle.getAttrData(characterId)
        characterSexId = characterData['Sex']
        characterSex = AttrText.getSexText(characterSexId)
        characterAge = characterData['Age']
        characterName = characterData['Name']
        fixCharacterName = TextLoading.getTextData(TextLoading.stageWordPath, '13')
        characterName = fixCharacterName + characterName
        attrListString.append(characterName)
        characterSelfName = characterData['SelfName']
        fixCharacterSelfName = TextLoading.getTextData(TextLoading.stageWordPath, '11')
        characterSelfName = fixCharacterSelfName + characterSelfName
        attrListString.append(characterSelfName)
        characterNickName = characterData['NickName']
        characterNickName = TextLoading.getTextData(TextLoading.stageWordPath, '12') + characterNickName
        attrListString.append(characterNickName)
        relationship = characterData['Relationship']
        relationship = TextLoading.getTextData(TextLoading.stageWordPath, '14') + relationship
        attrListString.append(relationship)
        characterSpecies = characterData['Species']
        characterSpecies = TextLoading.getTextData(TextLoading.stageWordPath, '15') + characterSpecies
        attrListString.append(characterSpecies)
        characterSex = TextLoading.getTextData(TextLoading.stageWordPath, '2') + characterSex
        attrListString.append(characterSex)
        characterAge = TextLoading.getTextData(TextLoading.stageWordPath, '3') + str(characterAge)
        attrListString.append(characterAge)
        EraPrint.p('\n')
        characterSan = characterData['San']
        characterSan = TextLoading.getTextData(TextLoading.stageWordPath, '10') + characterSan
        attrListString.append(characterSan)
        characterHeight = characterData['Height']['NowHeight']
        characterWeight = characterData['Weight']
        characterMeasurements = characterData['Measurements']
        characterHeightText = str(round(characterHeight,2))
        characterWeightText = str(round(characterWeight,2))
        characterBust = str(round(characterMeasurements['Bust'],2))
        characterWaist = str(round(characterMeasurements['Waist'],2))
        characterHip = str(round(characterMeasurements['Hip'],2))
        characterHeightInfo = TextLoading.getTextData(TextLoading.stageWordPath,'80') + characterHeightText
        attrListString.append(characterHeightInfo)
        characterWeightInfo = TextLoading.getTextData(TextLoading.stageWordPath,'81') + characterWeightText
        attrListString.append(characterWeightInfo)
        characterBustInfo = TextLoading.getTextData(TextLoading.stageWordPath,'82') + characterBust
        characterWaistInfo = TextLoading.getTextData(TextLoading.stageWordPath,'83') + characterWaist
        characterHipInfo = TextLoading.getTextData(TextLoading.stageWordPath,'84') + characterHip
        characterIntimate = characterData['Intimate']
        characterIntimate = TextLoading.getTextData(TextLoading.stageWordPath, '16') + characterIntimate
        attrListString.append(characterIntimate)
        characterGraces = characterData['Graces']
        characterGraces = TextLoading.getTextData(TextLoading.stageWordPath, '17') + characterGraces
        attrListString.append(characterGraces)
        attrListString.append(characterBustInfo)
        attrListString.append(characterWaistInfo)
        attrListString.append(characterHipInfo)
        EraPrint.plist(attrListString, 4, 'center')
        EraPrint.p('\n')
        AttrPrint.printHpAndMpBar(characterId)
        return 'CharacterMainAttrPanel'
    else:
        characterName = CacheContorl.characterData['character'][characterId]['Name']
        EraPrint.p(' ' + characterName + ' ')
        PyCmd.pcmd(panelStateOnText, 'CharacterMainAttrPanel', None)
        EraPrint.p('\n')
        return 'CharacterMainAttrPanel'

def seeCharacterEquipmentPanel(characterId:str) -> str:
    '''
    查看角色装备面板
    Keyword arguments:
    characterId -- 角色Id
    '''
    EraPrint.plittleline()
    EraPrint.p(TextLoading.getTextData(TextLoading.stageWordPath, '37'))
    panelState = CacheContorl.panelState['CharacterEquipmentPanel']
    if panelState == "0":
        PyCmd.pcmd(panelStateOffText,'CharacterEquipmentPanel')
        characterData = AttrHandle.getAttrData(characterId)
        EraPrint.p('\n')
        EraPrint.p(TextLoading.getTextData(TextLoading.stageWordPath, '39'))
        EraPrint.p('\n')
        characterClothingList = characterData['Clothing']
        characterClothingText = AttrText.getClothingText(characterClothingData)
        EraPrint.plist(characterClothingText, 4, 'center')
        EraPrint.p('\n')
        EraPrint.p(TextLoading.getTextData(TextLoading.stageWordPath, '40'))
        EraPrint.p('\n')
        characterSexItemList = characterData['SexItem']
        characterSexItemText = AttrText.getSexItemText(characterSexItemList)
        EraPrint.plist(characterSexItemText, 5, 'center')
    else:
        PyCmd.pcmd(panelStateOnText, 'CharacterEquipmentPanel', None)
        EraPrint.p('\n')
    return 'CharacterEquipmentPanel'

def seeCharacterItemPanel(characterId:str) -> str:
    '''
    查看角色携带道具面板
    Keyword arguments:
    characterId -- 角色Id
    '''
    EraPrint.plittleline()
    EraPrint.p(TextLoading.getTextData(TextLoading.stageWordPath, '38'))
    panelState = CacheContorl.panelState['CharacterItemPanel']
    if panelState == "0":
        PyCmd.pcmd(panelStateOffText, 'CharacterItemPanel')
        EraPrint.p('\n')
    else:
        PyCmd.pcmd(panelStateOnText, 'CharacterItemPanel')
        EraPrint.p('\n')
    return 'CharacterItemPanel'

def seeCharacterExperiencePanel(characterId:str) -> str:
    '''
    查看角色经验面板
    Keyword arguments:
    characterId -- 角色Id
    '''
    EraPrint.plittleline()
    EraPrint.p(TextLoading.getTextData(TextLoading.stageWordPath, '18'))
    panelState = CacheContorl.panelState['CharacterExperiencePanel']
    if panelState == "0":
        PyCmd.pcmd(panelStateOffText, 'CharacterExperiencePanel')
        characterData = AttrHandle.getAttrData(characterId)
        EraPrint.p('\n')
        characterSexExperienceList = characterData['SexExperience']
        characterSex = CacheContorl.characterData['character'][characterId]['Sex']
        characterSexTextList = AttrText.getSexExperienceText(characterSexExperienceList, characterSex)
        EraPrint.plist(characterSexTextList, 4, 'center')
    else:
        PyCmd.pcmd(panelStateOnText, 'CharacterExperiencePanel')
        EraPrint.p('\n')
    return 'CharacterExperiencePanel'

def seeCharacterLevelPanel(characterId:str) -> str:
    '''
    查看角色技能等级面板
    Keyword arguments:
    characterId -- 角色Id
    '''
    EraPrint.plittleline()
    EraPrint.p(TextLoading.getTextData(TextLoading.stageWordPath, '5'))
    panelState = CacheContorl.panelState['CharacterLevelPanel']
    if panelState == "0":
        PyCmd.pcmd(panelStateOffText, 'CharacterLevelPanel')
        EraPrint.p('\n')
        characterData = AttrHandle.getAttrData(characterId)
        characterSexGradeList = characterData['SexGrade']
        characterSex = CacheContorl.characterData['character'][characterId]['Sex']
        characterSexGradeTextList = AttrText.getSexGradeTextList(characterSexGradeList, characterSex)
        EraPrint.plist(characterSexGradeTextList, 4, 'center')
    else:
        PyCmd.pcmd(panelStateOnText, 'CharacterLevelPanel')
        EraPrint.p('\n')
    return 'CharacterLevelPanel'

def seeCharacterFeaturesPanel(characterId:str) -> str:
    '''
    查看角色特征面板
    Keyword arguments:
    characterId -- 角色Id
    '''
    EraPrint.plittleline()
    EraPrint.p(TextLoading.getTextData(TextLoading.stageWordPath, '6'))
    panelState = CacheContorl.panelState['CharacterFeaturesPanel']
    if panelState == "0":
        PyCmd.pcmd(panelStateOffText, 'CharacterFeaturesPanel')
        EraPrint.p('\n')
        characterData = AttrHandle.getAttrData(characterId)
        characterFeatures = characterData['Features']
        characterFeaturesStr = AttrText.getFeaturesStr(characterFeatures)
        EraPrint.p(characterFeaturesStr)
    else:
        PyCmd.pcmd(panelStateOnText, 'CharacterFeaturesPanel')
        EraPrint.p('\n')
    return 'CharacterFeaturesPanel'

def seeCharacterEngravingPanel(characterId:str) -> str:
    '''
    查看角色刻印面板
    Keyword arguments:
    characterId -- 角色Id
    '''
    EraPrint.plittleline()
    EraPrint.p(TextLoading.getTextData(TextLoading.stageWordPath, '7'))
    panelState = CacheContorl.panelState['CharacterEngravingPanel']
    if panelState == "0":
        PyCmd.pcmd(panelStateOffText, 'CharacterEngravingPanel')
        EraPrint.p('\n')
        characterData = AttrHandle.getAttrData(characterId)
        characterEngraving = characterData['Engraving']
        characterEngravingText = AttrText.getEngravingText(characterEngraving)
        EraPrint.plist(characterEngravingText, 3, 'center')
    else:
        PyCmd.pcmd(panelStateOnText, 'CharacterEngravingPanel')
        EraPrint.p('\n')
    return 'CharacterEngravingPanel'

def seeAttrShowHandlePanel() -> list:
    '''
    查看属性页显示控制
    '''
    ansListData = TextLoading.getTextData(TextLoading.cmdPath,'seeAttrPanelHandle')
    seeAttrPanelHandleCache = CacheContorl.panelState['AttrShowHandlePanel']
    inputS = []
    if seeAttrPanelHandleCache == '0':
        inputS.append(ansListData[2])
        inputS.append(ansListData[1])
    elif seeAttrPanelHandleCache == '1':
        inputS.append(ansListData[0])
        inputS.append(ansListData[2])
    elif seeAttrPanelHandleCache == '2':
        inputS.append(ansListData[1])
        inputS.append(ansListData[0])
    yrn = CmdButtonQueue.optionstr(CmdButtonQueue.seeattrpanelmenu, 2, cmdSize='center', askfor=False, cmdListData=inputS)
    return yrn

def askForSeeAttr() -> list:
    '''
    查看角色属性时输入处理面板
    '''
    yrn = CmdButtonQueue.optionint(CmdButtonQueue.seeattronrverytime, 3, cmdSize='center', askfor=False)
    return yrn

def inputAttrOverPanel():
    '''
    创建角色完成时确认角色属性输入处理面板
    '''
    yrn = CmdButtonQueue.optionint(CmdButtonQueue.acknowledgmentAttribute, askfor=False)
    return yrn
