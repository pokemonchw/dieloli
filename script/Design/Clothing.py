from script.Core import CacheContorl
import random

def creatorClothing(clothingName):
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
