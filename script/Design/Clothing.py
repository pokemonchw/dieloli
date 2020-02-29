from script.Core import CacheContorl,TextLoading,ValueHandle
import random
import math
import os
import datetime
import multiprocessing

clothingTagTextList = {
    "Sexy":TextLoading.getTextData(TextLoading.stageWordPath,'118'),
    "Handsome":TextLoading.getTextData(TextLoading.stageWordPath,'119'),
    "Elegant":TextLoading.getTextData(TextLoading.stageWordPath,'120'),
    "Fresh":TextLoading.getTextData(TextLoading.stageWordPath,'121'),
    "Sweet":TextLoading.getTextData(TextLoading.stageWordPath,'122'),
    "Warm":TextLoading.getTextData(TextLoading.stageWordPath,'123'),
    "Cleanliness":TextLoading.getTextData(TextLoading.stageWordPath,'124'),
}
clothingTypeTextList = {
    "Coat":TextLoading.getTextData(TextLoading.stageWordPath,'41'),
    "Underwear":TextLoading.getTextData(TextLoading.stageWordPath,'42'),
    "Pants":TextLoading.getTextData(TextLoading.stageWordPath,'43'),
    "Skirt":TextLoading.getTextData(TextLoading.stageWordPath,'44'),
    "Shoes":TextLoading.getTextData(TextLoading.stageWordPath,'45'),
    "Socks":TextLoading.getTextData(TextLoading.stageWordPath,'46'),
    "Bra":TextLoading.getTextData(TextLoading.stageWordPath,'47'),
    "Underpants":TextLoading.getTextData(TextLoading.stageWordPath,'48'),
}

def creatorSuit(suitName:str,sex:str) -> dict:
    '''
    创建套装
    Keyword arguments:
    suitName -- 套装模板
    sex -- 性别模板
    '''
    suitData = TextLoading.getTextData(TextLoading.equipmentPath,'Suit')[suitName][sex]
    newSuitData = {clothing:creatorClothing(suitData[clothing]) for clothing in suitData if suitData[clothing] != ''}
    return newSuitData

def creatorClothing(clothingName:str) -> dict:
    '''
    创建服装的基础函数
    Keyword arguments:
    clothingName -- 服装名字
    '''
    clothingData = {}
    clothingData['Sexy'] = random.randint(1,1000)
    clothingData['Handsome'] = random.randint(1,1000)
    clothingData['Elegant'] = random.randint(1,1000)
    clothingData['Fresh'] = random.randint(1,1000)
    clothingData['Sweet'] = random.randint(1,1000)
    clothingData['Warm'] = random.randint(0,30)
    clothingData['Price'] = sum([clothingData[x] for x in clothingData])
    setClothintEvaluationText(clothingData)
    clothingData['Cleanliness'] = 100
    clothingData.update(CacheContorl.clothingTypeData[clothingName])
    return clothingData

def getClothingNameData(clothingsData:dict) -> dict:
    '''
    按服装的具体名称对服装进行分类，获取同类下各服装的价值数据
    Keyword arguments:
    clothingsData -- 要分类的所有服装数据
    '''
    clothingNameData = {}
    for clothingType in clothingsData:
        clothingTypeData = clothingsData[clothingType]
        clothingNameData.setdefault(clothingType,{})
        for clothing in clothingTypeData:
            clothingData = clothingTypeData[clothing]
            clothingName = clothingData['Name']
            clothingNameData[clothingType].setdefault(clothingName,{})
            clothingNameData[clothingType][clothingName][clothing] = clothingData['Price'] + clothingData['Cleanliness']
        clothingNameData[clothingType] = {clothingName:ValueHandle.sortedDictForValues(clothingNameData[clothingType][clothingName]) for clothingName in clothingNameData[clothingType]}
    return clothingNameData

def getClothingPriceData(clothingsData:dict) -> dict:
    '''
    为每个类型的服装进行排序
    Keyword arguments:
    clothingsData -- 要排序的所有服装数据
    '''
    return {clothingType:{clothing:clothingsData[clothingType][clothing]['Price'] + clothingsData[clothingType][clothing]['Cleanliness'] for clothing in clothingsData[clothingType]} for clothingType in clothingsData}

def getClothingCollocationData(nowClothingData:dict,nowClothingType:str,clothingNameData:dict,clothingPriceData:dict,clothingData:dict):
    '''
    获取服装的当前搭配数据
    Keyword arguments:
    nowClothingData -- 当前服装原始数据
    nowClothingType -- 服装类型
    clothingNameData -- 按服装具体名字分类并按价值排序后的所有要搭配的服装数据
    clothingPriceData -- 按服装类型分类并按价值排序后的所有要搭配的服装数据
    clothingData -- 所有要查询的服装数据
    '''
    collocationData = {'Price':0}
    clothingCollocationTypeData = nowClothingData['CollocationalRestriction']
    for collocationType in clothingData:
        collocationData[collocationType] = ''
        if collocationType not in clothingCollocationTypeData:
            continue
        nowRestrict = clothingCollocationTypeData[collocationType]
        if nowRestrict == 'Precedence':
            clothingNowTypePrecedenceList = nowClothingData['Collocation'][collocationType]
            precedenceCollocation = getAppointNamesClothingTop(list(clothingNowTypePrecedenceList.keys()),clothingNameData[collocationType])
            if precedenceCollocation != 'None':
                collocationData[collocationType] = precedenceCollocation
                collocationData['Price'] += clothingPriceData[collocationType][precedenceCollocation]
            else:
                usuallyCollocation = getAppointTypeClothingTop(nowClothingData,nowClothingType,clothingData,clothingPriceData,collocationType,collocationData)
                if usuallyCollocation != 'None':
                    collocationData[collocationType] = usuallyCollocation
                    collocationData['Price'] += clothingPriceData[collocationType][usuallyCollocation]
                else:
                    collocationData = "None"
                    break
        elif nowRestrict == 'Usually':
            usuallyCollocation = getAppointTypeClothingTop(nowClothingData,nowClothingType,clothingData,clothingPriceData,collocationType,collocationData)
            if usuallyCollocation != 'None':
                collocationData[collocationType] = usuallyCollocation
                collocationData['Price'] += clothingPriceData[collocationType][usuallyCollocation]
            else:
                collocationData[collocationType] = ''
        elif nowRestrict == 'Must' or 'Ornone':
            clothingNowTypePrecedenceList = nowClothingData['Collocation'][collocationType]
            precedenceCollocation = getAppointNamesClothingTop(list(clothingNowTypePrecedenceList.keys()),clothingNameData[collocationType])
            if precedenceCollocation != 'None':
                collocationData[collocationType] = precedenceCollocation
                collocationData['Price'] += clothingPriceData[collocationType][precedenceCollocation]
            else:
                collocationData = 'None'
                break
    return collocationData

def getAppointNamesClothingTop(appointNameList:list,clothingTypeNameData:dict) -> str:
    '''
    获取指定服装类型数据下指定名称的服装中价值最高的服装
    Keyword arguments:
    appointNameList -- 要获取的服装名字列表
    clothingTypeNameData -- 以名字为分类的已排序的要查询的服装数据
    '''
    clothingData = {list(clothingTypeNameData[appoint].keys())[-1]:clothingTypeNameData[appoint][list(clothingTypeNameData[appoint].keys())[-1]] for appoint in appointNameList if appoint in clothingTypeNameData}
    if clothingData != {}:
        return list(ValueHandle.sortedDictForValues(clothingData).keys())[-1]
    return 'None'

def getAppointTypeClothingTop(nowClothingData:str,nowClothingType:str,clothingData:dict,clothingPriceData,newClothingType:str,collocationData:dict) -> str:
    '''
    获取指定类型下的可搭配的衣服中数值最高的衣服
    Keyword arguments:
    nowClothingName -- 当前服装名字
    nowClothingType -- 当前服装类型
    clothingData -- 要查询的所有服装数据
    clothingPriceData -- 已按价值排序的各类型服装数据
    newClothingType -- 要查询的服装类型
    collocationData -- 已有的穿戴数据
    '''
    clothingTypeData = clothingPriceData[newClothingType]
    clothingTypeDataList = list(clothingTypeData.keys())
    if clothingTypeDataList != []:
        clothingTypeDataList.reverse()
    for newClothing in clothingTypeDataList:
        newClothingData = clothingData[newClothingType][newClothing]
        returnJudge = True
        if judgeClothingCollocation(nowClothingData,nowClothingType,newClothingData,newClothingType) == False:
            continue
        for collocationType in collocationData:
            if collocationType == 'Price':
                continue
            nowCollocationId = collocationData[collocationType]
            if nowCollocationId == '':
                continue
            nowCollocationData = clothingData[collocationType][nowCollocationId]
            if judgeClothingCollocation(nowCollocationData,collocationType,newClothingData,newClothingType) == False:
                returnJudge = False
                break
        if returnJudge == False:
            continue
        return newClothing
    return 'None'

def judgeClothingCollocation(oldClothingData:dict,oldClothingType:str,newClothingData:dict,newClothingType:str) -> bool:
    '''
    判断两件服装是否能够进行搭配
    Keyword arguments:
    oldClothingData -- 旧服装数据
    oldClothingType -- 旧服装类型
    newClothingData -- 新服装数据
    newClothingType -- 新服装类型
    '''
    oldClothingDataRestrictData = oldClothingData['CollocationalRestriction']
    newClothingDataRestrictData = newClothingData['CollocationalRestriction']
    oldJudge = oldClothingDataRestrictData[newClothingType]
    newJudge = newClothingDataRestrictData[oldClothingType]
    if oldJudge in {'Must':0,'Ornone':1}:
        oldCollocationTypeData = oldClothingData['Collocation'][newClothingType]
        if newClothingData['Name'] not in oldCollocationTypeData:
            return False
    elif oldJudge == 'None':
        return False
    if newJudge in {'Must':0,'Ornone':1}:
        newCollocationTypeData = newClothingData['Collocation'][oldClothingType]
        if oldClothingData['Name'] not in newCollocationTypeData:
            return False
    elif newJudge == 'None':
        return False
    return True

clothingEvaluationTextList = [TextLoading.getTextData(TextLoading.stageWordPath,str(k)) for k in range(102,112)]
clothingTagList = [TextLoading.getTextData(TextLoading.stageWordPath,str(k)) for k in range(112,117)]
def setClothintEvaluationText(clothingData:dict):
    '''
    设置服装的评价文本
    Keyword arguments:
    clothingData -- 服装数据
    '''
    clothingAttrData = [clothingData['Sexy'],clothingData['Handsome'],clothingData['Elegant'],clothingData['Fresh'],clothingData['Sweet']]
    clothingAttrMax = sum(clothingAttrData)
    clothingEvaluationText = clothingEvaluationTextList[math.floor(clothingData['Price'] / 480) - 1]
    clothingTagText = clothingTagList[clothingAttrData.index(max(clothingAttrData))]
    clothingData['Evaluation'] = clothingEvaluationText
    clothingData['Tag'] = clothingTagText

def initCharacterClothingPutOn(playerPass=True):
    '''
    为所有角色穿衣服
    Keyword arguments:
    playerPass -- 跳过主角 (default:True)
    '''
    for character in CacheContorl.characterData['character']:
        if playerPass and character == 0:
            continue
        CacheContorl.characterData['character'][character].putOn()
