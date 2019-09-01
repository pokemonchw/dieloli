from script.Core import CacheContorl,TextLoading
import random
import math

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
    clothingData = CacheContorl.clothingTypeData[clothingName].copy()
    clothingData['Sexy'] = random.randint(1,1000)
    clothingData['Handsome'] = random.randint(1,1000)
    clothingData['Elegant'] = random.randint(1,1000)
    clothingData['Fresh'] = random.randint(1,1000)
    clothingData['Sweet'] = random.randint(1,1000)
    clothingData['Warm'] = random.randint(0,30)
    setClothintEvaluationText(clothingData)
    return clothingData

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
    clothingEvaluationText = clothingEvaluationTextList[math.floor(clothingAttrMax / 480)]
    clothingTagText = clothingTagList[clothingAttrData.index(max(clothingAttrData)) - 1]
    clothingData['Evaluation'] = clothingEvaluationText
    clothingData['Tag'] = clothingTagText
