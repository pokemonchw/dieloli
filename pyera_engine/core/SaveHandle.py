import os
try:
    import cPickle as pickle
except ImportError:
    import pickle
import core.EraPrint as eprint
import core.TextLoading as textload
import core.CacheContorl as cache
from core.pycfg import gamepath
import core.GameConfig as config

# 获取存档所在路径
def getSavefilePath(filename):
    filename = str(filename)
    savepath = os.path.join(gamepath,'save')
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    filepath = os.path.join(savepath,filename + '.save')
    return filepath

# 判断存档是否存在
def judgeSaveFileExist(saveId):
    savePath = getSavefilePath(saveId)
    if not os.path.exists(savePath):
        return "0"
    else:
        return "1"

# 存入存档数据
def establishSave(saveId):
    playerData = cache.playObject
    gameTime = cache.gameTime
    gameVerson = config.verson
    data = {"playerData":playerData,"gameTime":gameTime,"gameVerson":gameVerson}
    filepath = getSavefilePath(saveId)
    with open(filepath, 'wb') as f:
        pickle.dump(data,f)

# 读取存档数据
def loadSave(filename):
    filepath = getSavefilePath(filename)
    data = {}
    try:
        with open(filepath, 'rb') as f:
            data=pickle.load(f)
    except FileNotFoundError:
        eprint.p(textload.getTextData(textload.errorId,'notSaveError'))
    return data

# 确认存档读取
def inputLoadSave(saveId):
    saveData = loadSave(saveId)
    cache.playObject = saveData['playerData']
    cache.gameTime = saveData['gameTime']
    pass

# 获取存档页对应存档id
def getSavePageSaveId(pageSaveValue,inputId):
    savePanelPage = int(cache.panelState['SeeSaveListPanel']) + 1
    startSaveId = int(pageSaveValue) * (savePanelPage - 1)
    inputId = int(inputId)
    saveId = startSaveId + inputId
    return saveId

# 删除存档
def removeSave(saveId):
    savePath = getSavefilePath(saveId)
    if os.path.isfile(savePath):
        os.remove(savePath)
    else:
        errorText = textload.getTextData(textload.errorId,'notSaveError')
        eprint.pl(errorText)