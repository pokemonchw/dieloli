import core.CacheContorl as cache
import core.PyCmd as pycmd
import design.AttrCalculation as attr
import script.Panel.CreatorPlayerPanel as creatorplayerpanel
import core.TextLoading as textload
import script.flow.SeePlayerAttr as seeplayerattr
import core.EraPrint as eprint
import random
import core.ValueHandle as valuehandle

playerId = '0'
featuresList = attr.getFeaturesList()

# 请求玩家输入姓名流程
def inputName_func():
    cache.playObject['objectId'] = playerId
    flowReturn = creatorplayerpanel.inputNamePanel()
    if flowReturn == 0:
        pycmd.clr_cmd()
        inputNickName_func()
    elif flowReturn == 1:
        pycmd.clr_cmd()
        creatorplayerpanel.startInputNamePanel()
        pycmd.clr_cmd()
        inputName_func()
    elif flowReturn == 2:
        cache.wframeMouse['wFrameRePrint'] = 1
        eprint.pnextscreen()
        import design.mainflow as mainflow
        mainflow.main_func()
    pass

# 请求玩家输入昵称流程
def inputNickName_func():
    flowReturn = creatorplayerpanel.inputNickNamePanel()
    if flowReturn == 0:
        pycmd.clr_cmd()
        inputSelfName_func()
    elif flowReturn == 1:
        pycmd.clr_cmd()
        creatorplayerpanel.startInputNickNamePanel()
        pycmd.clr_cmd()
        inputNickName_func()
    elif flowReturn == 2:
        pycmd.clr_cmd()
        cache.temporaryObject['NickName'] = cache.temporaryObject['Name']
        inputNickName_func()
    elif flowReturn == 3:
        eprint.p('\n')
        pycmd.clr_cmd()
        inputName_func()
    pass

# 请求玩家输入自称流程
def inputSelfName_func():
    flowReturn = creatorplayerpanel.inputSelfNamePanel()
    if flowReturn == 0:
        pycmd.clr_cmd()
        inputSexConfirm_func()
    elif flowReturn == 1:
        pycmd.clr_cmd()
        creatorplayerpanel.startInputSelfName()
        pycmd.clr_cmd()
        inputSelfName_func()
    elif flowReturn == 2:
        eprint.p('\n')
        pycmd.clr_cmd()
        inputNickName_func()
    pass

# 请求玩家输入性别流程
def inputSexConfirm_func():
    flowReturn = creatorplayerpanel.inputSexPanel()
    sexId = cache.playObject['object'][playerId]['Sex']
    if flowReturn == 0:
        attr.setSexCache(sexId)
        if sexId == textload.getTextData(textload.roleId, 'Sex')[2]:
            cache.temporaryObject['Features']['Sex'] = textload.getTextData(textload.roleId, 'Features')['Sex'][0]
        elif sexId == textload.getTextData(textload.roleId, 'Sex')[3]:
            cache.temporaryObject['Features']['Sex'] = textload.getTextData(textload.roleId, 'Features')['Sex'][1]
        pycmd.clr_cmd()
        attributeGenerationBranch_func()
    elif flowReturn == 1:
        pycmd.clr_cmd()
        inputSexChoice_func()
    elif flowReturn == 2:
        pycmd.clr_cmd()
        inputNickName_func()
    pass

# 玩家确认性别流程
def inputSexChoice_func():
    sex = textload.getTextData(textload.roleId, 'Sex')
    sexMax = len(sex)
    flowReturn = creatorplayerpanel.inputSexChoicePanel()
    if flowReturn in range(0,sexMax):
        sexAtr = sex[flowReturn]
        cache.temporaryObject['Sex'] = sexAtr
        cache.playObject['object'][playerId] = cache.temporaryObject.copy()
        pycmd.clr_cmd()
        inputSexConfirm_func()
    elif flowReturn == 4:
        rand = random.randint(0, len(sex) - 1)
        sexAtr = sex[rand]
        cache.temporaryObject['Sex'] = sexAtr
        cache.playObject['object'][playerId] = cache.temporaryObject.copy()
        pycmd.clr_cmd()
        inputSexConfirm_func()
    elif flowReturn == 5:
        eprint.p('\n')
        pycmd.clr_cmd()
        inputSexConfirm_func()

# 询问玩家是否需要详细设置属性流程
def attributeGenerationBranch_func():
    flowReturn = creatorplayerpanel.attributeGenerationBranchPanel()
    if flowReturn == 0:
        pycmd.clr_cmd()
        detailedSetting_func1()
    elif flowReturn == 1:
        pycmd.clr_cmd()
        seeplayerattr.acknowledgmentAttribute_func()
    elif flowReturn == 2:
        pycmd.clr_cmd()
        inputSexConfirm_func()

# 详细设置属性1:询问玩家是否是小孩子
def detailedSetting_func1():
    flowRetun = creatorplayerpanel.detailedSetting1Panel()
    playerSex = cache.playObject['object']['0']['Sex']
    sexList = textload.getTextData(textload.roleId, 'Sex')
    if flowRetun == 0:
        pycmd.clr_cmd()
        detailedSetting_func2()
    elif flowRetun == 1:
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

# 详细设置属性2:询问玩家是否具备动物特征
def detailedSetting_func2():
    ansList = textload.getTextData(textload.cmdId, 'detailedSetting2')
    flowReturn = creatorplayerpanel.detailedSetting2Panel()
    if flowReturn == ansList[len(ansList)-1]:
        pycmd.clr_cmd()
        detailedSetting_func3()
    else:
        pycmd.clr_cmd()
        attr.setAnimalCache(flowReturn)
        detailedSetting_func3()

# 详细设置属性3:询问玩家是否具备丰富的性经验
def detailedSetting_func3():
    flowReturn = creatorplayerpanel.detailedSetting3Panel()
    sexTemDataList = valuehandle.dictKeysToList(textload.getTextData(textload.temId,'SexExperience'))
    sexTemDataList = valuehandle.reverseArrayList(sexTemDataList)
    sexTemName = sexTemDataList[flowReturn]
    if flowReturn != len(sexTemDataList) - 1:
        cache.featuresList['Chastity'] = ''
    else:
        pass
    playerSexExperienceData = attr.getSexExperience(sexTemName)
    cache.temporaryObject['SexExperience'] = playerSexExperienceData
    cache.temporaryObject['SexGrade'] = attr.getSexGrade(playerSexExperienceData)
    pycmd.clr_cmd()
    detailedSettind_func4()

# 详细设置属性4:询问玩家的胆量
def detailedSettind_func4():
    flowReturn = creatorplayerpanel.detailedSetting4Panel()
    courageList = featuresList['Courage']
    if flowReturn == 0:
        cache.featuresList['Courage'] = courageList[0]
    elif flowReturn == 1:
        pass
    elif flowReturn == 2:
        cache.featuresList['Courage'] = courageList[1]
    cache.temporaryObject['Features'] = cache.featuresList.copy()
    pycmd.clr_cmd()
    detailedSetting_func5()

# 详细设置属性5:询问玩家的性格
def detailedSetting_func5():
    flowReturn = creatorplayerpanel.detailedSetting5Panel()
    dispositionList = featuresList['Disposition']
    if flowReturn == 0:
        cache.featuresList['Disposition'] = dispositionList[0]
    elif flowReturn == 1:
        pass
    elif flowReturn == 2:
        cache.featuresList['Disposition'] = dispositionList[1]
    cache.temporaryObject['Features'] = cache.featuresList.copy()
    pycmd.clr_cmd()
    detailedSetting_func6()

# 详细设置属性6:询问玩家的自信
def detailedSetting_func6():
    flowReturn = creatorplayerpanel.detailedSetting6Panel()
    selfConfidenceList = featuresList['SelfConfidence']
    cache.featuresList['SelfConfidence'] = selfConfidenceList[flowReturn]
    cache.temporaryObject['Features'] = cache.featuresList.copy()
    pycmd.clr_cmd()
    detailedSetting_func7()

# 详细设置属性7:询问玩家友善
def detailedSetting_func7():
    flowReturn = creatorplayerpanel.detailedSetting7Panel()
    friendsList = featuresList['Friends']
    cache.featuresList['Friends'] = friendsList[flowReturn]
    cache.temporaryObject['Features'] = cache.featuresList.copy()
    pycmd.clr_cmd()
    seeplayerattr.acknowledgmentAttribute_func()