import random
from script.Core import CacheContorl,PyCmd,TextLoading,EraPrint,ValueHandle
from script.Design import AttrCalculation
from script.Panel import CreatorCharacterPanel
from script.Flow import SeeCharacterAttr

characterId = '0'
featuresList = AttrCalculation.getFeaturesList()

# 请求玩家输入姓名流程
def inputName_func():
    CacheContorl.characterData['characterId'] = characterId
    flowReturn = CreatorCharacterPanel.inputNamePanel()
    if flowReturn == 0:
        PyCmd.clr_cmd()
        inputNickName_func()
    elif flowReturn == 1:
        PyCmd.clr_cmd()
        CreatorCharacterPanel.startInputNamePanel()
        PyCmd.clr_cmd()
        inputName_func()
    elif flowReturn == 2:
        EraPrint.pnextscreen()
        CacheContorl.nowFlowId = 'title_frame'

# 请求玩家输入昵称流程
def inputNickName_func():
    flowReturn = CreatorCharacterPanel.inputNickNamePanel()
    if flowReturn == 0:
        PyCmd.clr_cmd()
        inputSelfName_func()
    elif flowReturn == 1:
        PyCmd.clr_cmd()
        CreatorCharacterPanel.startInputNickNamePanel()
        PyCmd.clr_cmd()
        inputNickName_func()
    elif flowReturn == 2:
        PyCmd.clr_cmd()
        CacheContorl.temporaryCharacter['NickName'] = CacheContorl.temporaryCharacter['Name']
        inputNickName_func()
    elif flowReturn == 3:
        PyCmd.clr_cmd()
        inputName_func()

# 请求玩家输入自称流程
def inputSelfName_func():
    flowReturn = CreatorCharacterPanel.inputSelfNamePanel()
    if flowReturn == 0:
        PyCmd.clr_cmd()
        inputSexConfirm_func()
    elif flowReturn == 1:
        PyCmd.clr_cmd()
        CreatorCharacterPanel.startInputSelfName()
        PyCmd.clr_cmd()
        inputSelfName_func()
    elif flowReturn == 2:
        EraPrint.p('\n')
        PyCmd.clr_cmd()
        inputNickName_func()

# 请求玩家输入性别流程
def inputSexConfirm_func():
    flowReturn = CreatorCharacterPanel.inputSexPanel()
    sexId = CacheContorl.characterData['character'][characterId]['Sex']
    if flowReturn == 0:
        AttrCalculation.setSexCache(sexId)
        sexKeysList = list(TextLoading.getTextData(TextLoading.rolePath,'Sex'))
        if sexId == sexKeysList[2]:
            CacheContorl.temporaryCharacter['Features']['Sex'] = TextLoading.getTextData(TextLoading.rolePath, 'Features')['Sex'][0]
        elif sexId == sexKeysList[3]:
            CacheContorl.temporaryCharacter['Features']['Sex'] = TextLoading.getTextData(TextLoading.rolePath, 'Features')['Sex'][1]
        PyCmd.clr_cmd()
        attributeGenerationBranch_func()
    elif flowReturn == 1:
        PyCmd.clr_cmd()
        inputSexChoice_func()
    elif flowReturn == 2:
        PyCmd.clr_cmd()
        inputNickName_func()

# 玩家确认性别流程
def inputSexChoice_func():
    sex = list(TextLoading.getTextData(TextLoading.rolePath, 'Sex').keys())
    sexMax = len(sex)
    flowReturn = CreatorCharacterPanel.inputSexChoicePanel()
    if flowReturn in range(0,sexMax):
        sexAtr = sex[flowReturn]
        CacheContorl.temporaryCharacter['Sex'] = sexAtr
        CacheContorl.characterData['character'][characterId] = CacheContorl.temporaryCharacter.copy()
        PyCmd.clr_cmd()
        inputSexConfirm_func()
    elif flowReturn == 4:
        rand = random.randint(0, len(sex) - 1)
        sexAtr = sex[rand]
        CacheContorl.temporaryCharacter['Sex'] = sexAtr
        CacheContorl.characterData['character'][characterId] = CacheContorl.temporaryCharacter.copy()
        PyCmd.clr_cmd()
        inputSexConfirm_func()
    elif flowReturn == 5:
        EraPrint.p('\n')
        PyCmd.clr_cmd()
        inputSexConfirm_func()

# 询问玩家是否需要详细设置属性流程
def attributeGenerationBranch_func():
    flowReturn = CreatorCharacterPanel.attributeGenerationBranchPanel()
    if flowReturn == 0:
        PyCmd.clr_cmd()
        detailedSetting_func1()
    elif flowReturn == 1:
        PyCmd.clr_cmd()
        CacheContorl.nowFlowId = 'acknowledgment_attribute'
    elif flowReturn == 2:
        PyCmd.clr_cmd()
        inputSexConfirm_func()

# 详细设置属性1:询问玩家是否是小孩子
def detailedSetting_func1():
    flowRetun = CreatorCharacterPanel.detailedSetting1Panel()
    characterSex = CacheContorl.characterData['character']['0']['Sex']
    sexList = list(TextLoading.getTextData(TextLoading.rolePath, 'Sex'))
    if flowRetun == 0:
        CacheContorl.featuresList['Age'] = featuresList["Age"][3]
    elif flowRetun == 4:
        if characterSex == sexList[0]:
            CacheContorl.featuresList['Age'] = featuresList["Age"][0]
        elif characterSex == sexList[1]:
            CacheContorl.featuresList['Age'] = featuresList["Age"][1]
        else:
            CacheContorl.featuresList['Age'] = featuresList["Age"][2]
        PyCmd.clr_cmd()
    CacheContorl.temporaryCharacter['Features'] = CacheContorl.featuresList.copy()
    characterAgeTemName = AttrCalculation.getAgeTemList()[flowRetun]
    characterAge = AttrCalculation.getAge(characterAgeTemName)
    characterTem = characterSex
    characterHeigt = AttrCalculation.getHeight(characterTem,characterAge,CacheContorl.temporaryCharacter['Features'])
    characterBmi = AttrCalculation.getBMI('Ordinary')
    characterWeight = AttrCalculation.getWeight(characterBmi,characterHeigt['NowHeight'])
    characterBodyFat = AttrCalculation.getBodyFat(characterSex,'Ordinary')
    characterMeasurements = AttrCalculation.getMeasurements(characterSex,characterHeigt['NowHeight'],characterWeight,characterBodyFat,'Ordinary')
    CacheContorl.temporaryCharacter['Age'] = characterAge
    CacheContorl.temporaryCharacter['Height'] = characterHeigt
    CacheContorl.temporaryCharacter['Weight'] = characterWeight
    CacheContorl.temporaryCharacter['BodyFat'] = characterBodyFat
    CacheContorl.temporaryCharacter['Measurements'] = characterMeasurements
    PyCmd.clr_cmd()
    detailedSetting_func2()

# 详细设置属性2:询问玩家是否具备动物特征
def detailedSetting_func2():
    ansList = TextLoading.getTextData(TextLoading.cmdPath, 'detailedSetting2')
    flowReturn = CreatorCharacterPanel.detailedSetting2Panel()
    if flowReturn == ansList[len(ansList)-1]:
        PyCmd.clr_cmd()
        detailedSetting_func3()
    else:
        PyCmd.clr_cmd()
        AttrCalculation.setAnimalCache(flowReturn)
        detailedSetting_func3()

# 详细设置属性3:询问玩家是否具备丰富的性经验
def detailedSetting_func3():
    flowReturn = CreatorCharacterPanel.detailedSetting3Panel()
    sexTemDataList = ValueHandle.dictKeysToList(TextLoading.getTextData(TextLoading.attrTemplatePath,'SexExperience'))
    sexTemDataList = ValueHandle.reverseArrayList(sexTemDataList)
    sexTemName = sexTemDataList[flowReturn]
    if flowReturn != len(sexTemDataList) - 1:
        CacheContorl.featuresList['Chastity'] = ''
    characterSexExperienceData = AttrCalculation.getSexExperience(sexTemName)
    CacheContorl.temporaryCharacter['SexExperience'] = characterSexExperienceData
    CacheContorl.temporaryCharacter['SexGrade'] = AttrCalculation.getSexGrade(characterSexExperienceData)
    PyCmd.clr_cmd()
    detailedSetting_func4()

# 详细设置属性4:询问玩家的胆量
def detailedSetting_func4():
    flowReturn = CreatorCharacterPanel.detailedSetting4Panel()
    courageList = featuresList['Courage']
    if flowReturn == 0:
        CacheContorl.featuresList['Courage'] = courageList[0]
    elif flowReturn == 2:
        CacheContorl.featuresList['Courage'] = courageList[1]
    CacheContorl.temporaryCharacter['Features'] = CacheContorl.featuresList.copy()
    PyCmd.clr_cmd()
    detailedSetting_func5()

# 详细设置属性5:询问玩家的性格
def detailedSetting_func5():
    flowReturn = CreatorCharacterPanel.detailedSetting5Panel()
    dispositionList = featuresList['Disposition']
    if flowReturn == 0:
        CacheContorl.featuresList['Disposition'] = dispositionList[0]
    elif flowReturn == 2:
        CacheContorl.featuresList['Disposition'] = dispositionList[1]
    CacheContorl.temporaryCharacter['Features'] = CacheContorl.featuresList.copy()
    PyCmd.clr_cmd()
    detailedSetting_func6()

# 详细设置属性6:询问玩家的自信
def detailedSetting_func6():
    flowReturn = CreatorCharacterPanel.detailedSetting6Panel()
    selfConfidenceList = featuresList['SelfConfidence']
    CacheContorl.featuresList['SelfConfidence'] = selfConfidenceList[flowReturn]
    CacheContorl.temporaryCharacter['Features'] = CacheContorl.featuresList.copy()
    PyCmd.clr_cmd()
    detailedSetting_func7()

# 详细设置属性7:询问玩家友善
def detailedSetting_func7():
    flowReturn = CreatorCharacterPanel.detailedSetting7Panel()
    friendsList = featuresList['Friends']
    CacheContorl.featuresList['Friends'] = friendsList[flowReturn]
    CacheContorl.temporaryCharacter['Features'] = CacheContorl.featuresList.copy()
    PyCmd.clr_cmd()
    detailedSetting_func8()

# 详细设置属性8:询问玩家体型
def detailedSetting_func8():
    flowReturn = CreatorCharacterPanel.detailedSetting8Panel()
    weightTemData = TextLoading.getTextData(TextLoading.attrTemplatePath,'WeightTem')
    weightTemList = ValueHandle.dictKeysToList(weightTemData)
    weightTem = weightTemList[int(flowReturn)]
    characterHeight = CacheContorl.temporaryCharacter['Height']
    characterBmi = AttrCalculation.getBMI(weightTem)
    characterWeight = AttrCalculation.getWeight(characterBmi, characterHeight['NowHeight'])
    characterSex = CacheContorl.temporaryCharacter['Sex']
    characterBodyFat = AttrCalculation.getBodyFat(characterSex,weightTem)
    characterMeasurements = AttrCalculation.getMeasurements(characterSex, characterHeight['NowHeight'], characterWeight,characterBodyFat,weightTem)
    CacheContorl.temporaryCharacter['Weight'] = characterWeight
    CacheContorl.temporaryCharacter['BodyFat'] = characterBodyFat
    CacheContorl.temporaryCharacter['Measurements'] = characterMeasurements
    CacheContorl.nowFlowId = 'acknowledgment_attribute'
