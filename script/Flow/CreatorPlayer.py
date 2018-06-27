import random
from script.Core import CacheContorl,PyCmd,TextLoading,EraPrint,ValueHandle
from script.Design import AttrCalculation
from script.Panel import CreatorPlayerPanel
from script.Flow import SeePlayerAttr

playerId = '0'
featuresList = AttrCalculation.getFeaturesList()

# 请求玩家输入姓名流程
def inputName_func():
    CacheContorl.playObject['objectId'] = playerId
    flowReturn = CreatorPlayerPanel.inputNamePanel()
    if flowReturn == 0:
        PyCmd.clr_cmd()
        inputNickName_func()
    elif flowReturn == 1:
        PyCmd.clr_cmd()
        CreatorPlayerPanel.startInputNamePanel()
        PyCmd.clr_cmd()
        inputName_func()
    elif flowReturn == 2:
        CacheContorl.wframeMouse['wFrameRePrint'] = 1
        EraPrint.pnextscreen()
        import Design.StartFlow as mainflow
        mainflow.main_func()
    pass

# 请求玩家输入昵称流程
def inputNickName_func():
    flowReturn = CreatorPlayerPanel.inputNickNamePanel()
    if flowReturn == 0:
        PyCmd.clr_cmd()
        inputSelfName_func()
    elif flowReturn == 1:
        PyCmd.clr_cmd()
        CreatorPlayerPanel.startInputNickNamePanel()
        PyCmd.clr_cmd()
        inputNickName_func()
    elif flowReturn == 2:
        PyCmd.clr_cmd()
        CacheContorl.temporaryObject['NickName'] = CacheContorl.temporaryObject['Name']
        inputNickName_func()
    elif flowReturn == 3:
        EraPrint.p('\n')
        PyCmd.clr_cmd()
        inputName_func()
    pass

# 请求玩家输入自称流程
def inputSelfName_func():
    flowReturn = CreatorPlayerPanel.inputSelfNamePanel()
    if flowReturn == 0:
        PyCmd.clr_cmd()
        inputSexConfirm_func()
    elif flowReturn == 1:
        PyCmd.clr_cmd()
        CreatorPlayerPanel.startInputSelfName()
        PyCmd.clr_cmd()
        inputSelfName_func()
    elif flowReturn == 2:
        EraPrint.p('\n')
        PyCmd.clr_cmd()
        inputNickName_func()
    pass

# 请求玩家输入性别流程
def inputSexConfirm_func():
    flowReturn = CreatorPlayerPanel.inputSexPanel()
    sexId = CacheContorl.playObject['object'][playerId]['Sex']
    if flowReturn == 0:
        AttrCalculation.setSexCache(sexId)
        if sexId == TextLoading.getTextData(TextLoading.roleId, 'Sex')[2]:
            CacheContorl.temporaryObject['Features']['Sex'] = TextLoading.getTextData(TextLoading.roleId, 'Features')['Sex'][0]
        elif sexId == TextLoading.getTextData(TextLoading.roleId, 'Sex')[3]:
            CacheContorl.temporaryObject['Features']['Sex'] = TextLoading.getTextData(TextLoading.roleId, 'Features')['Sex'][1]
        PyCmd.clr_cmd()
        attributeGenerationBranch_func()
    elif flowReturn == 1:
        PyCmd.clr_cmd()
        inputSexChoice_func()
    elif flowReturn == 2:
        PyCmd.clr_cmd()
        inputNickName_func()
    pass

# 玩家确认性别流程
def inputSexChoice_func():
    sex = TextLoading.getTextData(TextLoading.roleId, 'Sex')
    sexMax = len(sex)
    flowReturn = CreatorPlayerPanel.inputSexChoicePanel()
    if flowReturn in range(0,sexMax):
        sexAtr = sex[flowReturn]
        CacheContorl.temporaryObject['Sex'] = sexAtr
        CacheContorl.playObject['object'][playerId] = CacheContorl.temporaryObject.copy()
        PyCmd.clr_cmd()
        inputSexConfirm_func()
    elif flowReturn == 4:
        rand = random.randint(0, len(sex) - 1)
        sexAtr = sex[rand]
        CacheContorl.temporaryObject['Sex'] = sexAtr
        CacheContorl.playObject['object'][playerId] = CacheContorl.temporaryObject.copy()
        PyCmd.clr_cmd()
        inputSexConfirm_func()
    elif flowReturn == 5:
        EraPrint.p('\n')
        PyCmd.clr_cmd()
        inputSexConfirm_func()

# 询问玩家是否需要详细设置属性流程
def attributeGenerationBranch_func():
    flowReturn = CreatorPlayerPanel.attributeGenerationBranchPanel()
    if flowReturn == 0:
        PyCmd.clr_cmd()
        detailedSetting_func1()
    elif flowReturn == 1:
        PyCmd.clr_cmd()
        SeePlayerAttr.acknowledgmentAttribute_func()
    elif flowReturn == 2:
        PyCmd.clr_cmd()
        inputSexConfirm_func()

# 详细设置属性1:询问玩家是否是小孩子
def detailedSetting_func1():
    flowRetun = CreatorPlayerPanel.detailedSetting1Panel()
    playerSex = CacheContorl.playObject['object']['0']['Sex']
    sexList = TextLoading.getTextData(TextLoading.roleId, 'Sex')
    if flowRetun == 0:
        CacheContorl.featuresList['Age'] = featuresList["Age"][3]
    elif flowRetun == 4:
        if playerSex == sexList[0]:
            CacheContorl.featuresList['Age'] = featuresList["Age"][0]
        elif playerSex == sexList[1]:
            CacheContorl.featuresList['Age'] = featuresList["Age"][1]
        else:
            CacheContorl.featuresList['Age'] = featuresList["Age"][2]
        PyCmd.clr_cmd()
    CacheContorl.temporaryObject['Features'] = CacheContorl.featuresList.copy()
    playerAgeTemName = AttrCalculation.getAgeTemList()[flowRetun]
    playerAge = AttrCalculation.getAge(playerAgeTemName)
    playerTem = TextLoading.getTextData(TextLoading.temId,'TemList')[playerSex]
    playerHeigt = AttrCalculation.getHeight(playerTem,playerAge,CacheContorl.temporaryObject['Features'])
    playerWeight = AttrCalculation.getWeight('Ordinary',playerHeigt['NowHeight'])
    playerMeasurements = AttrCalculation.getMeasurements(playerTem,playerHeigt['NowHeight'],'Ordinary')
    CacheContorl.temporaryObject['Age'] = playerAge
    CacheContorl.temporaryObject['Height'] = playerHeigt
    CacheContorl.temporaryObject['Weight'] = playerWeight
    CacheContorl.temporaryObject['Measurements'] = playerMeasurements
    PyCmd.clr_cmd()
    detailedSetting_func2()

# 详细设置属性2:询问玩家是否具备动物特征
def detailedSetting_func2():
    ansList = TextLoading.getTextData(TextLoading.cmdId, 'detailedSetting2')
    flowReturn = CreatorPlayerPanel.detailedSetting2Panel()
    if flowReturn == ansList[len(ansList)-1]:
        PyCmd.clr_cmd()
        detailedSetting_func3()
    else:
        PyCmd.clr_cmd()
        AttrCalculation.setAnimalCache(flowReturn)
        detailedSetting_func3()

# 详细设置属性3:询问玩家是否具备丰富的性经验
def detailedSetting_func3():
    flowReturn = CreatorPlayerPanel.detailedSetting3Panel()
    sexTemDataList = ValueHandle.dictKeysToList(TextLoading.getTextData(TextLoading.temId,'SexExperience'))
    sexTemDataList = ValueHandle.reverseArrayList(sexTemDataList)
    sexTemName = sexTemDataList[flowReturn]
    if flowReturn != len(sexTemDataList) - 1:
        CacheContorl.featuresList['Chastity'] = ''
    else:
        pass
    playerSexExperienceData = AttrCalculation.getSexExperience(sexTemName)
    CacheContorl.temporaryObject['SexExperience'] = playerSexExperienceData
    CacheContorl.temporaryObject['SexGrade'] = AttrCalculation.getSexGrade(playerSexExperienceData)
    PyCmd.clr_cmd()
    detailedSettind_func4()

# 详细设置属性4:询问玩家的胆量
def detailedSettind_func4():
    flowReturn = CreatorPlayerPanel.detailedSetting4Panel()
    courageList = featuresList['Courage']
    if flowReturn == 0:
        CacheContorl.featuresList['Courage'] = courageList[0]
    elif flowReturn == 1:
        pass
    elif flowReturn == 2:
        CacheContorl.featuresList['Courage'] = courageList[1]
    CacheContorl.temporaryObject['Features'] = CacheContorl.featuresList.copy()
    PyCmd.clr_cmd()
    detailedSetting_func5()

# 详细设置属性5:询问玩家的性格
def detailedSetting_func5():
    flowReturn = CreatorPlayerPanel.detailedSetting5Panel()
    dispositionList = featuresList['Disposition']
    if flowReturn == 0:
        CacheContorl.featuresList['Disposition'] = dispositionList[0]
    elif flowReturn == 1:
        pass
    elif flowReturn == 2:
        CacheContorl.featuresList['Disposition'] = dispositionList[1]
    CacheContorl.temporaryObject['Features'] = CacheContorl.featuresList.copy()
    PyCmd.clr_cmd()
    detailedSetting_func6()

# 详细设置属性6:询问玩家的自信
def detailedSetting_func6():
    flowReturn = CreatorPlayerPanel.detailedSetting6Panel()
    selfConfidenceList = featuresList['SelfConfidence']
    CacheContorl.featuresList['SelfConfidence'] = selfConfidenceList[flowReturn]
    CacheContorl.temporaryObject['Features'] = CacheContorl.featuresList.copy()
    PyCmd.clr_cmd()
    detailedSetting_func7()

# 详细设置属性7:询问玩家友善
def detailedSetting_func7():
    flowReturn = CreatorPlayerPanel.detailedSetting7Panel()
    friendsList = featuresList['Friends']
    CacheContorl.featuresList['Friends'] = friendsList[flowReturn]
    CacheContorl.temporaryObject['Features'] = CacheContorl.featuresList.copy()
    PyCmd.clr_cmd()
    detailedSetting_func8()

# 详细设置属性8:询问玩家体型
def detailedSetting_func8():
    flowReturn = CreatorPlayerPanel.detailedSetting8Panel()
    weightTemData = TextLoading.getTextData(TextLoading.temId,'WeightTem')
    weightTemList = ValueHandle.dictKeysToList(weightTemData)
    weightTem = weightTemList[int(flowReturn)]
    playerHeight = CacheContorl.temporaryObject['Height']
    playerWeight = AttrCalculation.getWeight(weightTem, playerHeight['NowHeight'])
    playerSex = CacheContorl.temporaryObject['Sex']
    playerTem = TextLoading.getTextData(TextLoading.temId, 'TemList')[playerSex]
    playerMeasurements = AttrCalculation.getMeasurements(playerTem, playerHeight['NowHeight'], weightTem)
    CacheContorl.temporaryObject['Weight'] = playerWeight
    CacheContorl.temporaryObject['Measurements'] = playerMeasurements
    SeePlayerAttr.acknowledgmentAttribute_func()