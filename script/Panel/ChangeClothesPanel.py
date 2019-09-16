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
            clothingText = Clothing.clothingTypeTextList[clothingType] + clothingData['Evaluation'] + clothingData['Tag'] + clothingData['Name']
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
            drawText = Clothing.clothingTypeTextList[clothingText] + TextLoading.getTextData(TextLoading.stageWordPath,'117')
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
    for i in range(nowPageStartId,nowPageEndId):
        clothingId = list(characterClothingData.keys())[i]
        clothingData = characterClothingData[clothingId]
        clothingText = clothingData['Eevaluation'] = clothingData['Tag'] + clothingData['Name']
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
        idInfo = CmdButtonQueue.idIndex(i)
        cmdText = idInfo + drawText
        PyCmd.pcmd(cmdText,i,None)
        i += 1
        inputS.append(i)
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
    yrn = CmdButtonQueue.optionint(CmdButtonQueue.seecharacterwearclothes,idSize='center',askfor=False,startId=startId)
    return yrn
