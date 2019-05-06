import random,bisect

# 倒序排列表单
def reverseArrayList(list):
    ofList = []
    for i in reversed(list):
        ofList.append(i)
    return ofList

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

# 将str类型存储的数字list转换为int类型的list
def getListKeysIntList(nowList):
    newList = []
    for i in nowList:
        newList.append(int(i))
    return newList

# 按权重获取随机对象
def getRandomForWeight(data):
    weightMax = 0
    for i in data:
        weightMax += int(data[i])
    weightReginData = getReginList(data)
    weightReginList = getListKeysIntList(list(weightReginData.keys()))
    nowWeight = random.randint(0,weightMax - 1)
    weightRegin = getNextValueForList(nowWeight,weightReginList)
    return weightReginData[str(weightRegin)]

# 获取列表中第一个比当前值大的值
def getNextValueForList(nowInt,intList):
    nowId = bisect.bisect_left(intList,nowInt)
    return intList[nowId]
