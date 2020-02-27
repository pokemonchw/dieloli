import random
from script.Core import CacheContorl,PyCmd,TextLoading,EraPrint,ValueHandle,GameInit
from script.Design import AttrCalculation,GameTime,Nature,Character
from script.Panel import CreatorCharacterPanel,SeeNaturePanel

def inputName_func():
    '''
    请求玩家输入姓名流程
    输入0:进入请求玩家输入昵称流程(玩家姓名为默认或输入姓名流程结果)
    输入1:进入输入姓名流程
    输入2:返回标题菜单
    '''
    GameTime.initTime()
    CacheContorl.characterData['characterId'] = '0'
    CacheContorl.characterData['character']['0'] = Character.Character()
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

def inputNickName_func():
    '''
    请求玩家输入昵称流程
    输入0:进入请求玩家输入自称流程(玩家昵称为默认或输入玩家昵称流程结果)
    输入1:进入输入昵称流程
    输入2:使用玩家姓名作为昵称
    输入3:返回请求输入姓名流程
    '''
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
        CacheContorl.characterData['character']['0'].NickName = CacheContorl.characterData['character']['0'].Name
        inputNickName_func()
    elif flowReturn == 3:
        PyCmd.clr_cmd()
        inputName_func()

def inputSelfName_func():
    '''
    请求玩家输入自称流程
    输入0:进入请求玩家输入性别流程(玩家自称为默认，或输入自称流程结果)
    输入1:进入输入自称流程
    输入2:返回请求输入昵称流程
    '''
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

def inputSexConfirm_func():
    '''
    请求玩家输入性别流程
    输入0:进入询问是否进行详细设置流程(玩家性别为默认，或请求选择性别流程结果)
    输入1:进入选择性别流程
    输入2:返回请求输入自称流程
    '''
    flowReturn = CreatorCharacterPanel.inputSexPanel()
    sexId = CacheContorl.characterData['character']['0'].Sex
    if flowReturn == 0:
        PyCmd.clr_cmd()
        attributeGenerationBranch_func()
    elif flowReturn == 1:
        PyCmd.clr_cmd()
        inputSexChoice_func()
    elif flowReturn == 2:
        PyCmd.clr_cmd()
        inputSelfName_func()

def inputSexChoice_func():
    '''
    玩家选择性别流程
    输入0-3:选择对应性别(Man/Woman/Futa/Asexual)
    输入4:随机选择一个性别
    输入5:返回请求输入性别流程
    '''
    sex = list(TextLoading.getTextData(TextLoading.rolePath, 'Sex').keys())
    sexMax = len(sex)
    flowReturn = CreatorCharacterPanel.inputSexChoicePanel()
    if flowReturn in range(0,sexMax):
        sexAtr = sex[flowReturn]
        CacheContorl.characterData['character']['0'].Sex = sexAtr
        PyCmd.clr_cmd()
        inputSexConfirm_func()
    elif flowReturn == 4:
        rand = random.randint(0, len(sex) - 1)
        sexAtr = sex[rand]
        CacheContorl.characterData.characterData['character']['0'].Sex = sexAtr
        PyCmd.clr_cmd()
        inputSexConfirm_func()
    elif flowReturn == 5:
        EraPrint.p('\n')
        PyCmd.clr_cmd()
        inputSexConfirm_func()

def attributeGenerationBranch_func():
    '''
    询问玩家是否需要详细设置属性流程
    输入0:进入询问玩家年龄段流程
    输入1:进入属性最终确认流程(使用基础模板生成玩家属性)
    输入2:返回请求输入性别流程
    '''
    flowReturn = CreatorCharacterPanel.attributeGenerationBranchPanel()
    if flowReturn == 0:
        PyCmd.clr_cmd()
        detailedSetting_func1()
    elif flowReturn == 1:
        PyCmd.clr_cmd()
        CacheContorl.characterData['character']['0'].initAttr()
        CacheContorl.nowFlowId = 'acknowledgment_attribute'
    elif flowReturn == 2:
        PyCmd.clr_cmd()
        inputSexConfirm_func()

def detailedSetting_func1():
    '''
    询问玩家年龄模板流程
    '''
    flowRetun = CreatorCharacterPanel.detailedSetting1Panel()
    characterSex = CacheContorl.characterData['character']['0'].Sex
    sexList = list(TextLoading.getTextData(TextLoading.rolePath, 'Sex'))
    characterAgeTemName = AttrCalculation.getAgeTemList()[flowRetun]
    CacheContorl.characterData['character']['0'].Age = AttrCalculation.getAge(characterAgeTemName)
    PyCmd.clr_cmd()
    detailedSetting_func3()

def detailedSetting_func3():
    '''
    询问玩家性经验程度流程
    '''
    flowReturn = CreatorCharacterPanel.detailedSetting3Panel()
    sexTemDataList = list(TextLoading.getTextData(TextLoading.attrTemplatePath,'SexExperience').keys())
    sexTemDataList.reverse()
    sexTemName = sexTemDataList[flowReturn]
    characterSexExperienceData = AttrCalculation.getSexExperience(sexTemName)
    CacheContorl.characterData['character']['0'].SexExperienceTem = sexTemName
    PyCmd.clr_cmd()
    detailedSetting_func8()

def detailedSetting_func8():
    '''
    询问玩家肥胖程度流程
    '''
    flowReturn = CreatorCharacterPanel.detailedSetting8Panel()
    weightTemData = TextLoading.getTextData(TextLoading.attrTemplatePath,'WeightTem')
    weightTemList = list(weightTemData.keys())
    weightTem = weightTemList[int(flowReturn)]
    CacheContorl.characterData['character']['0'].WeigtTem = weightTem
    CacheContorl.characterData['character']['0'].BodyFatTem = weightTem
    enterCharacterNature_func()

def enterCharacterNature_func():
    '''
    请求玩家确认性格流程
    '''
    CacheContorl.characterData['character']['0'].initAttr()
    while 1:
        PyCmd.clr_cmd()
        CreatorCharacterPanel.enterCharacterNatureHead()
        inputS = SeeNaturePanel.seeCharacterNatureChangePanel('0')
        inputS += CreatorCharacterPanel.enterCharacterNatureEnd()
        yrn = GameInit.askfor_All(inputS)
        if yrn in CacheContorl.characterData['character']['0'].Nature:
            if CacheContorl.characterData['character']['0'].Nature[yrn] < 50:
                CacheContorl.characterData['character']['0'].Nature[yrn] = random.uniform(50,100)
            else:
                CacheContorl.characterData['character']['0'].Nature[yrn] = random.uniform(0,50)
        elif int(yrn) == 0:
            CacheContorl.characterData['character']['0'].initAttr()
            CacheContorl.nowFlowId = 'acknowledgment_attribute'
            break
        elif int(yrn) == 1:
            Nature.initCharacterNature('0')
