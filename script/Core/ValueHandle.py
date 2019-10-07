import random
import bisect

def twoBitArrayToDict(array:tuple) -> dict:
    '''
    将二维数组转换为字典
    Keyword arguments:
    array -- 要转换的二维数组
    '''
    newDict = dict((x, y) for x, y in array)
    return newDict

def getReginList(nowData:dict) -> dict:
    '''
    按dict中每个value的值对key进行排序，并计算权重区域列表
    Keyword arguments:
    nowData -- 需要进行计算权重的dict数据
    '''
    regionList = {}
    sortData = sortedDictForValues(nowData)
    sortDataKey = list(sortData.keys())
    regionIndex = 0
    for i in range(0,len(sortData)):
        nowKey = sortDataKey[i]
        regionIndex = regionIndex + sortData[nowKey]
        regionList[str(regionIndex)] = nowKey
    return regionList

def sortedDictForValues(oldDict):
    '''
    按dict中每个value的值对key进行排序生成新dict
    Keyword arguments:
    oldDict -- 需要进行排序的数据
    '''
    sortData = twoBitArrayToDict(sorted(oldDict.items(),key=lambda x:x[1]))
    return sortData

def getRandomForWeight(data:dict) -> 'dataKey':
    '''
    按权重随机获取dict中的一个key
    Keyword arguments:
    data -- 需要随机获取key的dict数据
    '''
    weightMax = sum(data.values())
    weightReginData = getReginList(data)
    weightReginList = list(map(int,weightReginData.keys()))
    nowWeight = random.randint(0,weightMax - 1)
    weightRegin = getNextValueForList(nowWeight,weightReginList)
    return weightReginData[str(weightRegin)]

def getNextValueForList(nowInt:int,intList:list) -> int:
    '''
    获取列表中第一个比指定值大的数
    Keyword arguments:
    nowInt -- 作为获取参考的指定数值
    intList -- 用于取值的列表
    '''
    nowId = bisect.bisect_left(intList,nowInt)
    return intList[nowId]

def getOldValueForList(nowInt:int,intList:list) -> int:
    '''
    获取列表中第一个比指定值小的数
    Keyword arguments:
    nowInt -- 作为获取参考的指定数值
    intList -- 用于取值的列表
    '''
    nowId = bisect.bisect_right(intList,nowInt)
    return intList[nowId - 1]
