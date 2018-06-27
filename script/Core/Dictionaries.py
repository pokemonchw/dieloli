from script.Core import CacheContorl

# 转义文本处理
def handleText(string):
    string = objectName(string)
    string = objectNickName(string)
    string = objectSelfName(string)
    return string

# 转义当前访问对象姓名
def objectName(string):
    try:
        objectId = CacheContorl.playObject['objectId']
        objectname = CacheContorl.playObject['object'][objectId]['Name']
        string = string.replace('{Name}', objectname)
        return string
    except KeyError:
        return string

# 转义当前访问对象昵称
def objectNickName(string):
    try:
        objectId = CacheContorl.playObject['objectId']
        objectname = CacheContorl.playObject['object'][objectId]['NickName']
        string = string.replace('{NickName}', objectname)
        return string
    except KeyError:
        return string

# 转义当前访问对象自称
def objectSelfName(string):
    try:
        objectId = CacheContorl.playObject['objectId']
        objectselfname = CacheContorl.playObject['object'][objectId]['SelfName']
        string = string.replace('{SelfName}',objectselfname)
        return string
    except KeyError:
        return string