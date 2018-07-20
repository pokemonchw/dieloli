from script.Core import CacheContorl,TextLoading,EraPrint,PyCmd
from script.Design import AttrPrint,AttrHandle,AttrText,CmdButtonQueue

panelStateTextData = TextLoading.getTextData(TextLoading.cmdId,'cmdSwitch')
panelStateOnText = panelStateTextData[1]
panelStateOffText = panelStateTextData[0]

# 初始化查看属性面板状态
def initShowAttrPanelList():
    CacheContorl.panelState['PlayerMainAttrPanel'] = '1'
    CacheContorl.panelState['PlayerEquipmentPanel'] = '1'
    CacheContorl.panelState['PlayerItemPanel'] = '1'
    CacheContorl.panelState['PlayerExperiencePanel'] = '1'
    CacheContorl.panelState['PlayerLevelPanel'] = '1'
    CacheContorl.panelState['PlayerFeaturesPanel'] = '1'
    CacheContorl.panelState['PlayerEngravingPanel'] = '1'

# 查看角色主属性面板
def seePlayerMainAttrPanel(playerId):
    title1 = TextLoading.getTextData(TextLoading.stageWordId, '1')
    EraPrint.plt(title1)
    playeridText = TextLoading.getTextData(TextLoading.stageWordId, '0') + playerId
    EraPrint.p(playeridText)
    panelState = CacheContorl.panelState['PlayerMainAttrPanel']
    if panelState == "0":
        PyCmd.pcmd(panelStateOffText,'PlayerMainAttrPanel',None)
        EraPrint.p('\n')
        attrListString = []
        playerData = AttrHandle.getAttrData(playerId)
        playerSex = playerData['Sex']
        playerAge = playerData['Age']
        playerName = playerData['Name']
        fixPlayerName = TextLoading.getTextData(TextLoading.stageWordId, '13')
        playerName = fixPlayerName + playerName
        attrListString.append(playerName)
        playerSelfName = playerData['SelfName']
        fixPlayerSelfName = TextLoading.getTextData(TextLoading.stageWordId, '11')
        playerSelfName = fixPlayerSelfName + playerSelfName
        attrListString.append(playerSelfName)
        playerNickName = playerData['NickName']
        playerNickName = TextLoading.getTextData(TextLoading.stageWordId, '12') + playerNickName
        attrListString.append(playerNickName)
        relationship = playerData['Relationship']
        relationship = TextLoading.getTextData(TextLoading.stageWordId, '14') + relationship
        attrListString.append(relationship)
        playerSpecies = playerData['Species']
        playerSpecies = TextLoading.getTextData(TextLoading.stageWordId, '15') + playerSpecies
        attrListString.append(playerSpecies)
        playerSex = TextLoading.getTextData(TextLoading.stageWordId, '2') + playerSex
        attrListString.append(playerSex)
        playerAge = TextLoading.getTextData(TextLoading.stageWordId, '3') + str(playerAge)
        attrListString.append(playerAge)
        EraPrint.p('\n')
        playerSan = playerData['San']
        playerSan = TextLoading.getTextData(TextLoading.stageWordId, '10') + playerSan
        attrListString.append(playerSan)
        playerHeight = playerData['Height']['NowHeight']
        playerWeight = playerData['Weight']
        playerMeasurements = playerData['Measurements']
        playerHeightText = str(round(playerHeight,2))
        playerWeightText = str(round(playerWeight,2))
        playerBust = str(round(playerMeasurements['Bust'],2))
        playerWaist = str(round(playerMeasurements['Waist'],2))
        playerHip = str(round(playerMeasurements['Hip'],2))
        playerHeightInfo = TextLoading.getTextData(TextLoading.stageWordId,'80') + playerHeightText
        attrListString.append(playerHeightInfo)
        playerWeightInfo = TextLoading.getTextData(TextLoading.stageWordId,'81') + playerWeightText
        attrListString.append(playerWeightInfo)
        playerBustInfo = TextLoading.getTextData(TextLoading.stageWordId,'82') + playerBust
        playerWaistInfo = TextLoading.getTextData(TextLoading.stageWordId,'83') + playerWaist
        playerHipInfo = TextLoading.getTextData(TextLoading.stageWordId,'84') + playerHip
        playerIntimate = playerData['Intimate']
        playerIntimate = TextLoading.getTextData(TextLoading.stageWordId, '16') + playerIntimate
        attrListString.append(playerIntimate)
        playerGraces = playerData['Graces']
        playerGraces = TextLoading.getTextData(TextLoading.stageWordId, '17') + playerGraces
        attrListString.append(playerGraces)
        attrListString.append(playerBustInfo)
        attrListString.append(playerWaistInfo)
        attrListString.append(playerHipInfo)
        EraPrint.plist(attrListString, 4, 'center')
        EraPrint.p('\n')
        AttrPrint.printHpAndMpBar(playerId)
        return 'PlayerMainAttrPanel'
    else:
        PyCmd.pcmd(panelStateOnText, 'PlayerMainAttrPanel', None)
        EraPrint.p('\n')
        return 'PlayerMainAttrPanel'


# 查看角色装备面板
def seePlayerEquipmentPanel(playerId):
    EraPrint.plittleline()
    EraPrint.p(TextLoading.getTextData(TextLoading.stageWordId, '37'))
    panelState = CacheContorl.panelState['PlayerEquipmentPanel']
    if panelState == "0":
        PyCmd.pcmd(panelStateOffText,'PlayerEquipmentPanel')
        playerData = AttrHandle.getAttrData(playerId)
        EraPrint.p('\n')
        EraPrint.p(TextLoading.getTextData(TextLoading.stageWordId, '39'))
        EraPrint.p('\n')
        playerClothingList = playerData['Clothing']
        playerClothingText = AttrText.getClothingText(playerClothingList)
        EraPrint.plist(playerClothingText, 4, 'center')
        EraPrint.p('\n')
        EraPrint.p(TextLoading.getTextData(TextLoading.stageWordId, '40'))
        EraPrint.p('\n')
        playerSexItemList = playerData['SexItem']
        playerSexItemText = AttrText.getSexItemText(playerSexItemList)
        EraPrint.plist(playerSexItemText, 5, 'center')
        return 'PlayerEquipmentPanel'
    else:
        PyCmd.pcmd(panelStateOnText, 'PlayerEquipmentPanel', None)
        EraPrint.p('\n')
        return 'PlayerEquipmentPanel'

# 查看角色携带道具面板
def seePlayerItemPanel(playerId):
    EraPrint.plittleline()
    EraPrint.p(TextLoading.getTextData(TextLoading.stageWordId, '38'))
    panelState = CacheContorl.panelState['PlayerItemPanel']
    if panelState == "0":
        PyCmd.pcmd(panelStateOffText, 'PlayerItemPanel')
        EraPrint.p('\n')
        return 'PlayerItemPanel'
    else:
        PyCmd.pcmd(panelStateOnText, 'PlayerItemPanel')
        EraPrint.p('\n')
        return 'PlayerItemPanel'
    pass

# 查看角色经验面板
def seePlayerExperiencePanel(playerId):
    EraPrint.plittleline()
    EraPrint.p(TextLoading.getTextData(TextLoading.stageWordId, '18'))
    panelState = CacheContorl.panelState['PlayerExperiencePanel']
    if panelState == "0":
        PyCmd.pcmd(panelStateOffText, 'PlayerExperiencePanel')
        playerData = AttrHandle.getAttrData(playerId)
        EraPrint.p('\n')
        playerSexExperienceList = playerData['SexExperience']
        playerSex = CacheContorl.playObject['object'][playerId]['Sex']
        playerSexTextList = AttrText.getSexExperienceText(playerSexExperienceList, playerSex)
        EraPrint.plist(playerSexTextList, 4, 'center')
        return 'PlayerExperiencePanel'
    else:
        PyCmd.pcmd(panelStateOnText, 'PlayerExperiencePanel')
        EraPrint.p('\n')
        return 'PlayerExperiencePanel'

# 查看角色技能等级
def seePlayerLevelPanel(playerId):
    EraPrint.plittleline()
    EraPrint.p(TextLoading.getTextData(TextLoading.stageWordId, '5'))
    panelState = CacheContorl.panelState['PlayerLevelPanel']
    if panelState == "0":
        PyCmd.pcmd(panelStateOffText, 'PlayerLevelPanel')
        EraPrint.p('\n')
        playerData = AttrHandle.getAttrData(playerId)
        playerSexGradeList = playerData['SexGrade']
        playerSex = CacheContorl.playObject['object'][playerId]['Sex']
        playerSexGradeTextList = AttrText.getSexGradeTextList(playerSexGradeList, playerSex)
        EraPrint.plist(playerSexGradeTextList, 4, 'center')
        return 'PlayerLevelPanel'
    else:
        PyCmd.pcmd(panelStateOnText, 'PlayerLevelPanel')
        EraPrint.p('\n')
        return 'PlayerLevelPanel'
    pass

# 查看角色特征
def seePlayerFeaturesPanel(playerId):
    EraPrint.plittleline()
    EraPrint.p(TextLoading.getTextData(TextLoading.stageWordId, '6'))
    panelState = CacheContorl.panelState['PlayerFeaturesPanel']
    if panelState == "0":
        PyCmd.pcmd(panelStateOffText, 'PlayerFeaturesPanel')
        EraPrint.p('\n')
        playerData = AttrHandle.getAttrData(playerId)
        playerFeatures = playerData['Features']
        playerFeaturesStr = AttrText.getFeaturesStr(playerFeatures)
        EraPrint.p(playerFeaturesStr)
        return 'PlayerFeaturesPanel'
    else:
        PyCmd.pcmd(panelStateOnText, 'PlayerFeaturesPanel')
        EraPrint.p('\n')
        return 'PlayerFeaturesPanel'
    pass

# 查看角色刻印
def seePlayerEngravingPanel(playerId):
    EraPrint.plittleline()
    EraPrint.p(TextLoading.getTextData(TextLoading.stageWordId, '7'))
    panelState = CacheContorl.panelState['PlayerEngravingPanel']
    if panelState == "0":
        PyCmd.pcmd(panelStateOffText, 'PlayerEngravingPanel')
        EraPrint.p('\n')
        playerData = AttrHandle.getAttrData(playerId)
        playerEngraving = playerData['Engraving']
        playerEngravingText = AttrText.getEngravingText(playerEngraving)
        EraPrint.plist(playerEngravingText, 3, 'center')
        return 'PlayerEngravingPanel'
    else:
        PyCmd.pcmd(panelStateOnText, 'PlayerEngravingPanel')
        EraPrint.p('\n')
        return 'PlayerEngravingPanel'

# 查看属性页显示控制
def seeAttrShowHandlePanel():
    ansListData = TextLoading.getTextData(TextLoading.cmdId,'seeAttrPanelHandle')
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

# 查看角色属性时输入面板
def askForSeeAttr():
    yrn = CmdButtonQueue.optionint(CmdButtonQueue.seeattronrverytime, 3, cmdSize='center', askfor=False)
    return yrn

# 创建角色完成时确认角色属性输入面板
def inputAttrOverPanel():
    yrn = CmdButtonQueue.optionint(CmdButtonQueue.acknowledgmentAttribute, 1, askfor=False)
    return yrn