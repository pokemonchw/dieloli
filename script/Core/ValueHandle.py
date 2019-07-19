import random,bisect,numpy,datetime

# 二维数组转字典
def twoBitArrayToDict(array):
    newDict = {}
    for i in range(0,len(array)):
        newDict[array[i][0]] = array[i][1]
    return newDict

# 按dict中每个key的Value进行排列,并算得权重域列表
def getReginList(nowData):
    regionList = {}
    sortData = sorted(nowData.items(),key=lambda x:x[1])
    sortData = twoBitArrayToDict(sortData)
    sortDataKey = list(sortData.keys())
    regionIndex = 0
    for i in range(0,len(sortData)):
        nowKey = sortDataKey[i]
        regionIndex = regionIndex + sortData[nowKey]
        regionList[str(regionIndex)] = nowKey
    return regionList

# 按权重获取随机对象
def getRandomForWeight(data):
    weightMax = sum(data.values())
    weightReginData = getReginList(data)
    weightReginList = list(map(int,weightReginData.keys()))
    nowWeight = random.randint(0,weightMax - 1)
    weightRegin = getNextValueForList(nowWeight,weightReginList)
    return weightReginData[str(weightRegin)]

# 获取列表中第一个比当前值大的值
def getNextValueForList(nowInt,intList):
    nowId = bisect.bisect_left(intList,nowInt)
    return intList[nowId]
