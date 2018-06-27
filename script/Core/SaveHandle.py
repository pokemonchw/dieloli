import os
try:
    import cPickle as pickle
except ImportError:
    import pickle
from script.Core import EraPrint,TextLoading,CacheContorl,GameConfig,GamePathConfig
from script.Design import MapHandle,CharacterHandle

gamepath = GamePathConfig.gamepath

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
    playerData = CacheContorl.playObject
    gameTime = CacheContorl.gameTime
    gameVerson = GameConfig.verson
    scaneData = CacheContorl.sceneData
    MapHandle.initScanePlayerData()
    data = {"playerData":playerData,"gameTime":gameTime,"gameVerson":gameVerson,"sceneData":scaneData}
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
        EraPrint.p(TextLoading.getTextData(TextLoading.errorId,'notSaveError'))
    return data

# 确认存档读取
def inputLoadSave(saveId):
    saveData = loadSave(saveId)
    CacheContorl.playObject = saveData['playerData']
    CacheContorl.playObject['objectId'] = '0'
    CacheContorl.gameTime = saveData['gameTime']
    CacheContorl.sceneData = saveData['sceneData']
    CharacterHandle.initPlayerPosition()
    pass

# 获取存档页对应存档id
def getSavePageSaveId(pageSaveValue,inputId):
    savePanelPage = int(CacheContorl.panelState['SeeSaveListPanel']) + 1
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
        errorText = TextLoading.getTextData(TextLoading.errorId,'notSaveError')
        EraPrint.pl(errorText)