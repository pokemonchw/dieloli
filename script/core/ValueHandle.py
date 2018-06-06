# 将一个列表中每一个元素逐个追加进另一个列表
def listAppendToList(ifList,ofList):
    ifListMax = len(ifList)
    for i in range(0,ifListMax):
        ofList.append(ifList[i])
    return ofList

# 将字典的键排序放入表单
def dictKeysToList(dict):
    value = []
    keys = dict.keys()
    for key in keys:
        value.append(key)
    return value

# 对字典的keys进行计数
def indexDictKeysMax(dict):
    list = dictKeysToList(dict)
    return len(list)

# 倒序排列表单
def reverseArrayList(list):
    ofList = []
    for i in reversed(list):
        ofList.append(i)
    return ofList
