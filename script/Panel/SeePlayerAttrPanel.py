import core.CacheContorl as cache
import script.AttrHandle as attrhandle
import core.TextLoading as textload
import core.EraPrint as eprint
import script.AttrPrint as attrprint
import script.AttrText as attrtext
import script.Ans as ans
import core.PyCmd as pycmd

panelStateTextData = textload.getTextData(textload.cmdId,'cmdSwitch')
panelStateOnText = panelStateTextData[1]
panelStateOffText = panelStateTextData[0]


# 初始化查看属性面板状态
def initShowAttrPanelList():
    cache.panelState['PlayerMainAttrPanel'] = '1'
    cache.panelState['PlayerEquipmentPanel'] = '1'
    cache.panelState['PlayerItemPanel'] = '1'
    cache.panelState['PlayerExperiencePanel'] = '1'
    cache.panelState['PlayerLevelPanel'] = '1'
    cache.panelState['PlayerFeaturesPanel'] = '1'
    cache.panelState['PlayerEngravingPanel'] = '1'

# 查看角色主属性面板
def seePlayerMainAttrPanel(playerId):
    title1 = textload.getTextData(textload.stageWordId, '1')
    eprint.plt(title1)
    playeridText = textload.getTextData(textload.stageWordId, '0') + playerId
    eprint.p(playeridText)
    panelState = cache.panelState['PlayerMainAttrPanel']
    if panelState == "0":
        pycmd.pcmd(panelStateOffText,'PlayerMainAttrPanel',None)
        eprint.p('\n')
        attrListString = []
        playerData = attrhandle.getAttrData(playerId)
        playerSex = playerData['Sex']
        playerAge = playerData['Age']
        playerName = playerData['Name']
        fixPlayerName = textload.getTextData(textload.stageWordId, '13')
        playerName = fixPlayerName + playerName
        attrListString.append(playerName)
        playerSelfName = playerData['SelfName']
        fixPlayerSelfName = textload.getTextData(textload.stageWordId, '11')
        playerSelfName = fixPlayerSelfName + playerSelfName
        attrListString.append(playerSelfName)
        playerNickName = playerData['NickName']
        playerNickName = textload.getTextData(textload.stageWordId, '12') + playerNickName
        attrListString.append(playerNickName)
        relationship = playerData['Relationship']
        relationship = textload.getTextData(textload.stageWordId, '14') + relationship
        attrListString.append(relationship)
        playerSpecies = playerData['Species']
        playerSpecies = textload.getTextData(textload.stageWordId, '15') + playerSpecies
        attrListString.append(playerSpecies)
        playerSex = textload.getTextData(textload.stageWordId, '2') + playerSex
        attrListString.append(playerSex)
        playerAge = textload.getTextData(textload.stageWordId, '3') + str(playerAge)
        attrListString.append(playerAge)
        eprint.p('\n')
        playerSan = playerData['San']
        playerSan = textload.getTextData(textload.stageWordId, '10') + playerSan
        attrListString.append(playerSan)
        playerIntimate = playerData['Intimate']
        playerIntimate = textload.getTextData(textload.stageWordId, '16') + playerIntimate
        attrListString.append(playerIntimate)
        playerGraces = playerData['Graces']
        playerGraces = textload.getTextData(textload.stageWordId, '17') + playerGraces
        attrListString.append(playerGraces)
        eprint.plist(attrListString, 4, 'center')
        eprint.p('\n')
        attrprint.printHpAndMpBar(playerId)
        return 'PlayerMainAttrPanel'
    else:
        pycmd.pcmd(panelStateOnText, 'PlayerMainAttrPanel', None)
        eprint.p('\n')
        return 'PlayerMainAttrPanel'


# 查看角色装备面板
def seePlayerEquipmentPanel(playerId):
    eprint.plittleline()
    eprint.p(textload.getTextData(textload.stageWordId, '37'))
    panelState = cache.panelState['PlayerEquipmentPanel']
    if panelState == "0":
        pycmd.pcmd(panelStateOffText,'PlayerEquipmentPanel')
        playerData = attrhandle.getAttrData(playerId)
        eprint.p('\n')
        eprint.p(textload.getTextData(textload.stageWordId, '39'))
        eprint.p('\n')
        playerClothingList = playerData['Clothing']
        print(playerData)
        playerClothingText = attrtext.getClothingText(playerClothingList)
        eprint.plist(playerClothingText, 4, 'center')
        eprint.p('\n')
        eprint.p(textload.getTextData(textload.stageWordId, '40'))
        eprint.p('\n')
        playerSexItemList = playerData['SexItem']
        playerSexItemText = attrtext.getSexItemText(playerSexItemList)
        eprint.plist(playerSexItemText, 5, 'center')
        return 'PlayerEquipmentPanel'
    else:
        pycmd.pcmd(panelStateOnText, 'PlayerEquipmentPanel', None)
        eprint.p('\n')
        return 'PlayerEquipmentPanel'

# 查看角色携带道具面板
def seePlayerItemPanel(playerId):
    eprint.plittleline()
    eprint.p(textload.getTextData(textload.stageWordId, '38'))
    panelState = cache.panelState['PlayerItemPanel']
    if panelState == "0":
        pycmd.pcmd(panelStateOffText, 'PlayerItemPanel')
        eprint.p('\n')
        return 'PlayerItemPanel'
    else:
        pycmd.pcmd(panelStateOnText, 'PlayerItemPanel')
        eprint.p('\n')
        return 'PlayerItemPanel'
    pass

# 查看角色经验面板
def seePlayerExperiencePanel(playerId):
    eprint.plittleline()
    eprint.p(textload.getTextData(textload.stageWordId, '18'))
    panelState = cache.panelState['PlayerExperiencePanel']
    if panelState == "0":
        pycmd.pcmd(panelStateOffText, 'PlayerExperiencePanel')
        playerData = attrhandle.getAttrData(playerId)
        eprint.p('\n')
        playerSexExperienceList = playerData['SexExperience']
        playerSexTextList = attrtext.getSexExperienceText(playerSexExperienceList,
                                                          cache.playObject['object']['0']['Sex'])
        eprint.plist(playerSexTextList, 4, 'center')
        return 'PlayerExperiencePanel'
    else:
        pycmd.pcmd(panelStateOnText, 'PlayerExperiencePanel')
        eprint.p('\n')
        return 'PlayerExperiencePanel'

# 查看角色技能等级
def seePlayerLevelPanel(playerId):
    eprint.plittleline()
    eprint.p(textload.getTextData(textload.stageWordId, '5'))
    panelState = cache.panelState['PlayerLevelPanel']
    if panelState == "0":
        pycmd.pcmd(panelStateOffText, 'PlayerLevelPanel')
        eprint.p('\n')
        playerData = attrhandle.getAttrData(playerId)
        playerSexGradeList = playerData['SexGrade']
        playerSexGradeTextList = attrtext.getSexGradeTextList(playerSexGradeList,
                                                              cache.playObject['object']['0']['Sex'])
        eprint.plist(playerSexGradeTextList, 4, 'center')
        return 'PlayerLevelPanel'
    else:
        pycmd.pcmd(panelStateOnText, 'PlayerLevelPanel')
        eprint.p('\n')
        return 'PlayerLevelPanel'
    pass

# 查看角色特征
def seePlayerFeaturesPanel(playerId):
    eprint.plittleline()
    eprint.p(textload.getTextData(textload.stageWordId, '6'))
    panelState = cache.panelState['PlayerFeaturesPanel']
    if panelState == "0":
        pycmd.pcmd(panelStateOffText, 'PlayerFeaturesPanel')
        eprint.p('\n')
        playerData = attrhandle.getAttrData(playerId)
        playerFeatures = playerData['Features']
        playerFeaturesStr = attrtext.getFeaturesStr(playerFeatures)
        eprint.p(playerFeaturesStr)
        return 'PlayerFeaturesPanel'
    else:
        pycmd.pcmd(panelStateOnText, 'PlayerFeaturesPanel')
        eprint.p('\n')
        return 'PlayerFeaturesPanel'
    pass

# 查看角色刻印
def seePlayerEngravingPanel(playerId):
    eprint.plittleline()
    eprint.p(textload.getTextData(textload.stageWordId, '7'))
    panelState = cache.panelState['PlayerEngravingPanel']
    if panelState == "0":
        pycmd.pcmd(panelStateOffText, 'PlayerEngravingPanel')
        eprint.p('\n')
        playerData = attrhandle.getAttrData(playerId)
        playerEngraving = playerData['Engraving']
        playerEngravingText = attrtext.getEngravingText(playerEngraving)
        eprint.plist(playerEngravingText, 3, 'center')
        return 'PlayerEngravingPanel'
    else:
        pycmd.pcmd(panelStateOnText, 'PlayerEngravingPanel')
        eprint.p('\n')
        return 'PlayerEngravingPanel'

# 查看属性页显示控制
def seeAttrShowHandlePanel():
    ansListData = textload.getTextData(textload.cmdId,'seeAttrPanelHandle')
    seeAttrPanelHandleCache = cache.panelState['AttrShowHandlePanel']
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
    yrn = ans.optionstr(ans.seeattrpanelmenu,2,cmdSize='center',askfor=False,cmdListData=inputS)
    return yrn

# 查看角色属性时输入面板
def askForSeeAttr():
    yrn = ans.optionint(ans.seeattronrverytime,3,cmdSize='center',askfor=False)
    return yrn

# 创建角色完成时确认角色属性输入面板
def inputAttrOverPanel():
    yrn = ans.optionint(ans.acknowledgmentAttribute, 1,askfor=False)
    return yrn