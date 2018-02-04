import core.EraPrint as eprint
import script.TextLoading as textload
import core.game as game
import core.CacheContorl as cache
import script.Ans as ans
import core.PyCmd as pycmd
import random
import script.AttrCalculation as attr
import core.TextHandle as text
import script.ProportionalBar as proportionalBar
import core.GameConfig as config
import script.AttrText as attrtext
import script.flow.MainFrame as mainframe

playerId = '0'
featuresList = attr.getFeaturesList()

# 请求玩家输入姓名界面
def inputName_func():
    cache.playObject['objectId'] = playerId
    cache.playObject['object'][playerId] = cache.temporaryObject.copy()
    attr.setDefaultCache()
    eprint.p('\n')
    eprint.pline()
    eprint.pl(textload.loadMessageAdv('4'))
    yrn = ans.optionint(ans.currencymenu,1)
    eprint.p('\n')
    if yrn == 0:
        pycmd.clr_cmd()
        inputNickName_func()
        return
    elif yrn == 1:
        pycmd.clr_cmd()
        eprint.pline()
        eprint.pl(textload.loadMessageAdv('3'))
        inputState = 0
        while inputState == 0:
            playerName = game.askfor_str()
            eprint.pl(playerName)
            if text.getTextIndex(playerName) > 10:
                eprint.pl(textload.loadErrorText('inputNameTooLongError'))
            else:
                inputState = 1
                cache.temporaryObject['Name'] = playerName
        eprint.p('\n')
        pycmd.clr_cmd()
        inputName_func()
    elif yrn == 2:
        cache.wframeMouse['wFrameRePrint'] = 1
        eprint.pnextscreen()
        import script.mainflow as mainflow
        mainflow.main_func()
    pass

# 请求玩家输入昵称界面
def inputNickName_func():
    cache.playObject['object'][playerId] = cache.temporaryObject.copy()
    eprint.pline()
    eprint.pl(textload.loadMessageAdv('6'))
    yrn = ans.optionint(ans.inputnickname,1)
    eprint.p('\n')
    if yrn == 0:
        pycmd.clr_cmd()
        inputSelfName_func()
    elif yrn == 1:
        pycmd.clr_cmd()
        eprint.pline()
        eprint.pl(textload.loadMessageAdv('5'))
        inputState = 0
        while inputState == 0:
            playerNickName = game.askfor_str()
            eprint.pl(playerNickName)
            if text.getTextIndex(playerNickName) > 10:
                eprint.pl(textload.loadErrorText('inputNickNameTooLongError'))
            else:
                inputState = 1
                cache.temporaryObject['NickName'] = playerNickName
        eprint.p('\n')
        pycmd.clr_cmd()
        inputNickName_func()
    elif yrn == 2:
        pycmd.clr_cmd()
        cache.temporaryObject['NickName'] = cache.temporaryObject['Name']
        inputNickName_func()
    elif yrn == 3:
        pycmd.clr_cmd()
        eprint.p('\n')
        inputName_func()
    pass

# 请求玩家输入自称界面
def inputSelfName_func():
    pycmd.clr_cmd()
    cache.playObject['object'][playerId] = cache.temporaryObject.copy()
    eprint.pline()
    eprint.pl(textload.loadMessageAdv('14'))
    yrn = ans.optionint(ans.inputselfname,1)
    eprint.p('\n')
    if yrn == 0:
        pycmd.clr_cmd()
        inputSexConfirm_func()
    elif yrn == 1:
        pycmd.clr_cmd()
        eprint.pline()
        eprint.pl(textload.loadMessageAdv('15'))
        inputState = 0
        while inputState == 0:
            playerSelfName = game.askfor_str()
            eprint.pl(playerSelfName)
            if text.getTextIndex(playerSelfName) > 10:
                eprint.pl(textload.loadErrorText('inputNickNameTooLongError'))
            else:
                inputState = 1
                cache.temporaryObject['SelfName'] = playerSelfName
        eprint.p('\n')
        pycmd.clr_cmd()
        inputSelfName_func()
    elif yrn == 2:
        pycmd.clr_cmd()
        eprint.p('\n')
        inputNickName_func()

# 请求玩家输入性别界面
def inputSexConfirm_func():
    pycmd.clr_cmd()
    sexId = cache.playObject['object'][playerId]['Sex']
    eprint.pline()
    eprint.pl(textload.loadMessageAdv('8')[sexId])
    yrn = ans.optionint(ans.currencymenu,1)
    eprint.p('\n')
    if yrn == 0:
        attr.setSexCache(sexId)
        if sexId == textload.loadRoleAtrText('Sex')[2]:
            cache.temporaryObject['Features']['Sex'] = textload.loadRoleAtrText('Features')['Sex'][0]
        elif sexId == textload.loadRoleAtrText('Sex')[3]:
            cache.temporaryObject['Features']['Sex'] = textload.loadRoleAtrText('Features')['Sex'][1]
        pycmd.clr_cmd()
        attributeGenerationBranch_func()
    elif yrn == 1:
        pycmd.clr_cmd()
        inputSexChoice_func()
    elif yrn == 2:
        pycmd.clr_cmd()
        inputNickName_func()
    pass

# 玩家确认性别界面
def inputSexChoice_func():
    pycmd.clr_cmd()
    eprint.pline()
    eprint.pl(textload.loadMessageAdv('7'))
    yrn = ans.optionint(ans.sexmenu,1)
    eprint.p('\n')
    sex = textload.loadRoleAtrText('Sex')
    sexMax = len(sex)
    if yrn in range(0,sexMax):
        sexAtr = sex[yrn]
        cache.temporaryObject['Sex'] = sexAtr
        cache.playObject['object'][playerId] = cache.temporaryObject.copy()
        pycmd.clr_cmd()
        inputSexConfirm_func()
    elif yrn == 4:
        rand = random.randint(0, len(sex) - 1)
        sexAtr = sex[rand]
        cache.temporaryObject['Sex'] = sexAtr
        cache.playObject['object'][playerId] = cache.temporaryObject.copy()
        pycmd.clr_cmd()
        inputSexConfirm_func()
    elif yrn == 5:
        pycmd.clr_cmd()
        eprint.p('\n')
        inputSexConfirm_func()
    pass

# 询问玩家是否需要详细设置属性
def attributeGenerationBranch_func():
    playerSex = cache.playObject['object']['0']['Sex']
    temlist = attr.getTemList()
    temId = temlist[playerSex]
    temData = attr.getAttr(temId)
    cache.temporaryObject['Age'] = temData['Age']
    cache.temporaryObject['SexExperience'] = temData['SexExperienceList']
    cache.temporaryObject['SexGrade'] = temData['SexGradeList']
    cache.temporaryObject['Engraving'] = temData['EngravingList']
    cache.temporaryObject['Features'] = cache.featuresList.copy()
    pycmd.clr_cmd()
    eprint.pline()
    eprint.pl(textload.loadMessageAdv('9'))
    yrn = ans.optionint(ans.currencymenu,1)
    if yrn == 0:
        pycmd.clr_cmd()
        detailedSetting_func1()
    elif yrn == 1:
        pycmd.clr_cmd()
        acknowledgmentAttribute_func()
    elif yrn == 2:
        pycmd.clr_cmd()
        inputSexConfirm_func()
    pass

# 详细设置属性1:询问玩家是否是小孩子
def detailedSetting_func1():
    eprint.p('\n')
    eprint.pline()
    playerSex = cache.playObject['object']['0']['Sex']
    sexList = textload.loadRoleAtrText('Sex')
    eprint.pl(textload.loadMessageAdv('10'))
    yrn = ans.optionint(ans.detailedsetting1,1)
    if yrn == 0:
        pycmd.clr_cmd()
        detailedSetting_func2()
    elif yrn == 1:
        if playerSex == sexList[0]:
            cache.featuresList['Age'] = featuresList["Age"][0]
        elif playerSex == sexList[1]:
            cache.featuresList['Age'] = featuresList["Age"][1]
        else:
            cache.featuresList['Age'] = featuresList["Age"][2]
        pycmd.clr_cmd()
        cache.temporaryObject['Features'] = cache.featuresList.copy()
        playerAgeTemName = attr.getAgeTemList()[1]
        playerAge = attr.getAge(playerAgeTemName)
        cache.temporaryObject['Age'] = playerAge
        detailedSetting_func2()
    pass

# 详细设置属性2:询问玩家是否具备动物特征
def detailedSetting_func2():
    eprint.p('\n')
    eprint.pline()
    eprint.pl(textload.loadMessageAdv('11'))
    ansList = textload.loadCmdAdv("detailedSetting2")
    yrn = ans.optionstr(ans.detailedsetting2,5,'center',True)
    if yrn == ansList[len(ansList)-1]:
        pycmd.clr_cmd()
        detailedSetting_func3()
    else:
        pycmd.clr_cmd()
        attr.setAnimalCache(yrn)
        detailedSetting_func3()
    pass

# 详细设置属性3:询问玩家是否具备丰富的性经验
def detailedSetting_func3():
    eprint.p('\n')
    eprint.pline()
    eprint.pl(textload.loadMessageAdv('12'))
    yrn = ans.optionint(ans.detailedsetting3)
    if not yrn == 3:
        cache.featuresList['Chastity'] = ''
    else:
        pass
    sexTemDataList = textload.loadAttrTemplateText('SexExperience')['SexTemList']
    sexTemName = sexTemDataList[yrn]
    playerSexExperienceData = attr.getSexExperience(sexTemName)
    cache.temporaryObject['SexExperience'] = playerSexExperienceData
    cache.temporaryObject['SexGrade'] = attr.getSexGrade(playerSexExperienceData)
    pycmd.clr_cmd()
    detailedSettind_func4()

# 详细设置属性4:询问玩家的胆量
def detailedSettind_func4():
    eprint.p('\n')
    eprint.pline()
    eprint.pl(textload.loadMessageAdv('13'))
    yrn = ans.optionint(ans.detailedsetting4)
    courageList = featuresList['Courage']
    if yrn == 0:
        cache.featuresList['Courage'] = courageList[0]
    elif yrn == 1:
        pass
    elif yrn == 2:
        cache.featuresList['Courage'] = courageList[1]
    cache.temporaryObject['Features'] = cache.featuresList.copy()
    pycmd.clr_cmd()
    detailedSetting_func5()

# 详细设置属性5:询问玩家的性格
def detailedSetting_func5():
    eprint.p('\n')
    eprint.pline()
    eprint.pl(textload.loadMessageAdv('16'))
    yrn = ans.optionint(ans.detailedsetting5)
    dispositionList = featuresList['Disposition']
    if yrn == 0:
        cache.featuresList['Disposition'] = dispositionList[0]
    elif yrn == 1:
        pass
    elif yrn == 2:
        cache.featuresList['Disposition'] = dispositionList[1]
    cache.temporaryObject['Features'] = cache.featuresList.copy()
    pycmd.clr_cmd()
    detailedSetting_func6()

# 详细设置属性6:询问玩家的自信
def detailedSetting_func6():
    eprint.p('\n')
    eprint.pline()
    eprint.pl(textload.loadMessageAdv('17'))
    yrn = ans.optionint(ans.detailedsetting6)
    selfConfidenceList = featuresList['SelfConfidence']
    cache.featuresList['SelfConfidence'] = selfConfidenceList[yrn]
    cache.temporaryObject['Features'] = cache.featuresList.copy()
    pycmd.clr_cmd()
    detailedSetting_func7()

# 详细设置属性7:询问玩家友善
def detailedSetting_func7():
    eprint.p('\n')
    eprint.pline()
    eprint.pl(textload.loadMessageAdv('18'))
    yrn = ans.optionint(ans.detailedsetting7)
    friendsList = featuresList['Friends']
    cache.featuresList['Friends'] = friendsList[yrn]
    cache.temporaryObject['Features'] = cache.featuresList.copy()
    pycmd.clr_cmd()
    acknowledgmentAttribute_func()

# 确认玩家属性界面
def acknowledgmentAttribute_func():
    attrListString = []
    cache.playObject['object']['0'] = cache.temporaryObject.copy()
    playerSex = cache.playObject['object']['0']['Sex']
    playerAge = cache.playObject['object']['0']['Age']
    title1 = textload.loadStageWordText('1')
    playerName = cache.playObject['object']['0']['Name']
    eprint.plt(title1)
    playerid = textload.loadStageWordText('0') + '0'
    eprint.p(playerid)
    fixPlayerName = textload.loadStageWordText('13')
    playerName = fixPlayerName + playerName
    attrListString.append(playerName)
    playerSelfName = cache.playObject['object']['0']['SelfName']
    fixPlayerSelfName = textload.loadStageWordText('11')
    playerSelfName = fixPlayerSelfName + playerSelfName
    attrListString.append(playerSelfName)
    playerNickName = cache.playObject['object']['0']['NickName']
    playerNickName = textload.loadStageWordText('12') + playerNickName
    attrListString.append(playerNickName)
    relationship = cache.playObject['object']['0']['Relationship']
    relationship = textload.loadStageWordText('14') + relationship
    attrListString.append(relationship)
    playerSpecies = cache.playObject['object']['0']['Species']
    playerSpecies = textload.loadStageWordText('15') + playerSpecies
    attrListString.append(playerSpecies)
    playerSex = textload.loadStageWordText('2') + playerSex
    attrListString.append(playerSex)
    playerAge = textload.loadStageWordText('3') + str(playerAge)
    attrListString.append(playerAge)
    eprint.p('\n')
    playerSan = cache.playObject['object']['0']['San']
    playerSan = textload.loadStageWordText('10') + playerSan
    attrListString.append(playerSan)
    playerIntimate = cache.playObject['object']['0']['Intimate']
    playerIntimate = textload.loadStageWordText('16') + playerIntimate
    attrListString.append(playerIntimate)
    playerGraces = cache.playObject['object']['0']['Graces']
    playerGraces = textload.loadStageWordText('17') + playerGraces
    attrListString.append(playerGraces)
    eprint.plist(attrListString,4,'center')
    eprint.p('\n')
    playerHitPoint = cache.playObject['object']['0']['HitPoint']
    playerMaxHitPoint = cache.playObject['object']['0']['HitPointMax']
    hitPointText = textload.loadStageWordText('8')
    hitPointBar = proportionalBar.getProportionalBar(hitPointText,playerMaxHitPoint,playerHitPoint,'hp','❤')
    textWidth = config.text_width
    indexHitPointBar = text.getTextIndex(hitPointBar)
    fixHitPointBar = ' ' * int((int(textWidth/2) - int(indexHitPointBar))/2)
    hitPointBar = hitPointBar
    eprint.p(fixHitPointBar)
    eprint.p(hitPointBar)
    playerManaPoint = cache.playObject['object']['0']['ManaPoint']
    playerMaxManaPoint = cache.playObject['object']['0']['ManaPointMax']
    manaPointText = textload.loadStageWordText('9')
    manaPointBar = proportionalBar.getProportionalBar(manaPointText,playerMaxManaPoint,playerManaPoint,'mp','❤')
    eprint.p(fixHitPointBar)
    eprint.p(manaPointBar)
    eprint.p('\n')
    eprint.plittleline()
    eprint.p(textload.loadStageWordText('37'))
    eprint.p('\n')
    eprint.p(textload.loadStageWordText('39'))
    eprint.p('\n')
    eprint.p(textload.loadStageWordText('40'))
    eprint.plittleline()
    eprint.p(textload.loadStageWordText('38'))
    eprint.p('\n')
    eprint.plittleline()
    eprint.p(textload.loadStageWordText('18'))
    eprint.p('\n')
    playerSexExperienceList = cache.playObject['object']['0']['SexExperience']
    playerSexTextList = attrtext.getSexExperienceText(playerSexExperienceList,cache.playObject['object']['0']['Sex'])
    eprint.plist(playerSexTextList,4,'center')
    eprint.plittleline()
    eprint.pl(textload.loadStageWordText('5'))
    playerSexGradeList = cache.playObject['object']['0']['SexGrade']
    playerSexGradeTextList = attrtext.getSexGradeTextList(playerSexGradeList,cache.playObject['object']['0']['Sex'])
    eprint.plist(playerSexGradeTextList,4,'center')
    eprint.plittleline()
    eprint.pl(textload.loadStageWordText('6'))
    playerFeatures = cache.playObject['object']['0']['Features']
    playerFeaturesStr = attrtext.getFeaturesStr(playerFeatures)
    eprint.p(playerFeaturesStr)
    eprint.plittleline()
    eprint.pl(textload.loadStageWordText('7'))
    playerEngraving = cache.playObject['object']['0']['Engraving']
    playerEngravingText = attrtext.getEngravingText(playerEngraving)
    eprint.plist(playerEngravingText,4,'center')
    eprint.p('\n')
    eprint.pline()
    yrn = ans.optionint(ans.acknowledgmentAttribute,1)
    if yrn == 0:
        pycmd.clr_cmd()
        mainframe.mainFrame_func()
    elif yrn == 1:
        cache.wframeMouse['wFrameRePrint'] = 1
        eprint.pnextscreen()
        import script.mainflow as mainflow
        mainflow.main_func()
    pass