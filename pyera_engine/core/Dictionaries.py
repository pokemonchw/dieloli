import core.CacheContorl as cache

def handleText(string):
    string = objectName(string)
    return string

def objectName(string):
    try:
        objectId = cache.playObject['objectId']
        objectname = cache.playObject['object'][objectId]['Name']
        string = string.replace('{Name}', objectname)
        return string
    except KeyError:
        return string