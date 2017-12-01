import core.CacheContorl as cache

def handleText(string):
    string = objectName(string)
    string = objectNickName(string)
    return string

def objectName(string):
    try:
        objectId = cache.playObject['objectId']
        objectname = cache.playObject['object'][objectId]['Name']
        string = string.replace('{Name}', objectname)
        return string
    except KeyError:
        return string

def objectNickName(string):
    try:
        objectId = cache.playObject['objectId']
        objectname = cache.playObject['object'][objectId]['NickName']
        string = string.replace('{NickName}', objectname)
        return string
    except KeyError:
        return string
