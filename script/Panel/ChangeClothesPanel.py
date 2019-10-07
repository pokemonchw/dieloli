from script.Core import EraPrint,TextLoading,CacheContorl,PyCmd,TextHandle,GameConfig
from script.Design import AttrText,CmdButtonQueue,Clothing

def seeCharacterWearClothesInfo(characterId:str):
    '''
    查看角色已穿戴服装列表顶部面板
    Keyword arguments:
    characterId -- 角色id
    '''
    sceneInfo = TextLoading.getTextData(TextLoading.stageWordPath,'79')
    EraPrint.plt(sceneInfo)
    characterInfo = AttrText.getCharacterAbbreviationsInfo(characterId)
    EraPrint.p(characterInfo)

def seeCharacterWearClothes(characterId:str,changeButton:bool):
    '''
    查看角色穿戴服装列表面板
    Keyword arguments:
    characterId -- 角色id
    changeButton -- 将服装列表绘制成按钮的开关
    '''
    EraPrint.p('\n')
    characterClothingData = CacheContorl.characterData['character'][characterId]['Clothing']
    characterPutOnList = CacheContorl.characterData['character'][characterId]['PutOn']
    clothingTextData = {}
    tagTextIndex = 0
    for i in range(len(Clothing.clothingTypeTextList.keys())):
        clothingType = list(Clothing.clothingTypeTextList.keys())[i]
        clothingId = characterPutOnList[clothingType]
        if clothingId == '':
            clothingTextData[clothingType] = 'None'
        else:
            clothingData = characterClothingData[clothingType][clothingId]
            clothingText = Clothing.clothingTypeTextList[clothingType] + ':' + clothingData['Evaluation'] + clothingData['Tag'] + clothingData['Name']
            clothingTextData[clothingText] = {}
            for tag in Clothing.clothingTagTextList:
                tagText = Clothing.clothingTagTextList[tag] + str(clothingData[tag])
                clothingTextData[clothingText][tagText] = 0
                nowTagTextIndex = TextHandle.getTextIndex(tagText)
                if nowTagTextIndex > tagTextIndex:
                    tagTextIndex = nowTagTextIndex
    longClothingTextIndex = TextHandle.getTextIndex(max(clothingTextData.keys(),key=TextHandle.getTextIndex))
    i = 0
    inputS = []
    for clothingText in clothingTextData:
        drawText = ''
        EraPrint.plittleline()
        if clothingTextData[clothingText] == 'None':
            drawText = Clothing.clothingTypeTextList[clothingText] + ':' + TextLoading.getTextData(TextLoading.stageWordPath,'117')
        else:
            nowClothingTextIndex = TextHandle.getTextIndex(clothingText)
            drawText += clothingText + ' '
            if nowClothingTextIndex < longClothingTextIndex:
                drawText += ' ' * (longClothingTextIndex - nowClothingTextIndex)
            for tagText in clothingTextData[clothingText]:
                nowTagTextIndex = TextHandle.getTextIndex(tagText)
                if nowTagTextIndex < tagTextIndex:
                    drawText += ' ' + tagText + ' ' * (tagTextIndex - nowTagTextIndex)
                else:
                    drawText += ' ' + tagText
        if changeButton:
            idInfo = CmdButtonQueue.idIndex(i)
            cmdText = idInfo + drawText
            PyCmd.pcmd(cmdText,i,None)
        else:
            EraPrint.p(drawText)
        i += 1
        inputS.append(str(i))
        EraPrint.p('\n')
    return inputS

def seeCharacterWearClothesCmd(startId:int) -> str:
    '''
    用于控制查看角色已装备服装列表面板的命令菜单
    '''
    EraPrint.pline()
    yrn = CmdButtonQueue.optionint(CmdButtonQueue.seecharacterwearclothes,idSize='center',askfor=False,startId=startId)
    return yrn

def seeCharacterClothesPanel(characterId:str,clothingType:str,maxPage:int):
    '''
    用于查看角色服装列表的面板
    Keyword arguments:
    characterId -- 角色id
    clothingType -- 服装类型
    maxPage -- 服装列表最大页数
    '''
    EraPrint.p('\n')
    characterClothingData = CacheContorl.characterData['character'][characterId]['Clothing'][clothingType]
    characterPutOnList = CacheContorl.characterData['character'][characterId]['PutOn']
    clothingTextData = {}
    tagTextIndex = 0
    nowPageId = int(CacheContorl.panelState["SeeCharacterClothesPanel"])
    nowPageMax = GameConfig.see_character_clothes_max
    nowPageStartId = nowPageId * nowPageMax
    nowPageEndId = nowPageStartId + nowPageMax
    if characterClothingData == {}:
        EraPrint.p(TextLoading.getTextData(TextLoading.messagePath,'34'))
        EraPrint.p('\n')
        return []
    if nowPageEndId > len(characterClothingData.keys()):
        nowPageEndId = len(characterClothingData.keys())
    passId = None
    for i in range(nowPageStartId,nowPageEndId):
        clothingId = list(characterClothingData.keys())[i]
        if clothingId == CacheContorl.characterData['character'][characterId]['PutOn'][clothingType]:
            passId = i - nowPageStartId
        clothingData = characterClothingData[clothingId]
        clothingText = clothingData['Evaluation'] + clothingData['Tag'] + clothingData['Name']
        clothingTextData[clothingText] = {}
        for tag in Clothing.clothingTagTextList:
            tagText = Clothing.clothingTagTextList[tag] + str(clothingData[tag])
            clothingTextData[clothingText][tagText] = 0
            nowTagTextIndex = TextHandle.getTextIndex(tagText)
            if nowTagTextIndex == nowTagTextIndex:
                tagTextIndex = nowTagTextIndex
    longClothingTextIndex = TextHandle.getTextIndex(max(clothingTextData.keys(),key=TextHandle.getTextIndex))
    i = 0
    inputS = []
    for clothingText in clothingTextData:
        drawText = ''
        EraPrint.plittleline()
        nowClothingTextIndex = TextHandle.getTextIndex(clothingText)
        drawText += clothingText + ' '
        if nowClothingTextIndex < longClothingTextIndex:
            drawText += ' ' * (longClothingTextIndex - nowClothingTextIndex)
        for tagText in clothingTextData[clothingText]:
            nowTagTextIndex = TextHandle.getTextIndex(tagText)
            if nowTagTextIndex < tagTextIndex:
                drawText += ' ' + tagText + ' ' * (tagTextIndex - nowTagTextIndex)
            else:
                drawText += ' ' + tagText
        if i == passId:
            drawText += ' ' + TextLoading.getTextData(TextLoading.stageWordPath,'125')
        idInfo = CmdButtonQueue.idIndex(i)
        cmdText = idInfo + drawText
        inputS.append(str(i))
        PyCmd.pcmd(cmdText,i,None)
        i += 1
    EraPrint.p('\n')
    pageText = '(' + str(nowPageId) + '/' + str(maxPage) + ')'
    EraPrint.printPageLine(sample='-',string=pageText)
    EraPrint.p('\n')
    return inputS

def seeCharacterClothesInfo(characterId:str):
    '''
    查看角色服装列表顶部面板
    Keyword arguments:
    characterId -- 角色id
    '''
    sceneInfo = TextLoading.getTextData(TextLoading.stageWordPath,'101')
    EraPrint.plt(sceneInfo)
    characterInfo = AttrText.getCharacterAbbreviationsInfo(characterId)
    EraPrint.p(characterInfo)

def seeCharacterWearClothesCmd(startId:int) -> str:
    '''
    用于控制查看角色服装列表面板的命令菜单
    '''
    EraPrint.pline()
    yrn = CmdButtonQueue.optionint(CmdButtonQueue.seecharacterwearclothes,cmdSize='center',askfor=False,startId=startId)
    return yrn

def seeCharacterClothesCmd(startId:int,nowClothingType:str) -> str:
    '''
    用于控制查看角色服装列表面板的命令菜单
    Keyword arguments:
    startId -- cmd命令的初始Id
    nowClothingType -- 当前列表的服装类型
    '''
    EraPrint.pline()
    clothingTypeList = list(Clothing.clothingTypeTextList.keys())
    cmdList = TextLoading.getTextData(TextLoading.cmdPath,CmdButtonQueue.seecharacterclothes)
    nowClothingTypeIndex = clothingTypeList.index(nowClothingType)
    upTypeId = nowClothingTypeIndex - 1
    if nowClothingTypeIndex == 0:
        upTypeId = len(clothingTypeList) - 1
    nextTypeId = nowClothingTypeIndex + 1
    if nowClothingTypeIndex == len(clothingTypeList) - 1:
        nextTypeId = 0
    upTypeText = [Clothing.clothingTypeTextList[clothingTypeList[upTypeId]]]
    nextTypeText = [Clothing.clothingTypeTextList[clothingTypeList[nextTypeId]]]
    cmdList = upTypeText + cmdList + nextTypeText
    yrn = CmdButtonQueue.optionint(None,5,cmdSize='center',askfor=False,startId=startId,cmdListData=cmdList)
    return yrn

def askSeeClothingInfoPanel(wearClothingJudge:bool) -> str:
    '''
    用于询问查看或穿戴服装的面板
    Keyword arguments:
    wearClothingJudge -- 当前服装穿戴状态
    '''
    EraPrint.p('\n')
    EraPrint.pline()
    titileMessage = TextLoading.getTextData(TextLoading.messagePath,'35')
    cmdData = TextLoading.getTextData(TextLoading.cmdPath,CmdButtonQueue.askseeclothinginfopanel).copy()
    if wearClothingJudge:
        del cmdData['0']
    else:
        del cmdData['1']
    cmdList = list(cmdData.values())
    return CmdButtonQueue.optionint(None,cmdListData=cmdList)

def seeClothingInfoPanel(characterId:str,clothingType:str,clothingId:str,wearClothingJudge:bool):
    '''
    查看服装详细信息面板
    Keyword arguments:
    characterId -- 角色id
    clothingType -- 服装类型
    clothingId -- 服装id
    '''
    EraPrint.plt(TextLoading.getTextData(TextLoading.stageWordPath,'126'))
    clothingData = CacheContorl.characterData['character'][characterId]['Clothing'][clothingType][clothingId]
    infoList = []
    clothingName = clothingData['Name']
    if wearClothingJudge:
        clothingName += ' ' + TextLoading.getTextData(TextLoading.stageWordPath,'125')
    infoList.append(TextLoading.getTextData(TextLoading.stageWordPath,'128') + clothingName)
    clothingTypeText = Clothing.clothingTypeTextList[clothingType]
    infoList.append(TextLoading.getTextData(TextLoading.stageWordPath,'129') + clothingTypeText)
    evaluationText = TextLoading.getTextData(TextLoading.stageWordPath,'131') + clothingData['Evaluation']
    infoList.append(evaluationText)
    EraPrint.plist(infoList,3,'center')
    EraPrint.sontitleprint(TextLoading.getTextData(TextLoading.stageWordPath,'130'))
    tagTextList = []
    for tag in Clothing.clothingTagTextList:
        tagText = Clothing.clothingTagTextList[tag]
        tagText += str(clothingData[tag])
        tagTextList.append(tagText)
    EraPrint.plist(tagTextList,4,'center')
    EraPrint.sontitleprint(TextLoading.getTextData(TextLoading.stageWordPath,'127'))
    EraPrint.p(clothingData['Describe'])

def seeClothingInfoAskPanel(wearClothingJudge:bool) -> str:
    '''
    查看服装详细信息的控制面板
    Keyword arguments:
    wearClothingJudge -- 服装穿戴状态
    '''
    EraPrint.pline()
    cmdData = TextLoading.getTextData(TextLoading.cmdPath,CmdButtonQueue.seeclothinginfoaskpanel).copy()
    if wearClothingJudge:
        del cmdData['1']
    else:
        del cmdData['2']
    cmdList = list(cmdData.values())
    return CmdButtonQueue.optionint(None,4,cmdSize='center',cmdListData=cmdList)
