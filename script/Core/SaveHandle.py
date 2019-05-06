import os,pickle
from script.Core import EraPrint,TextLoading,CacheContorl,GameConfig,GamePathConfig
from script.Design import CharacterHandle

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
    return "1"

# 存入存档数据
def establishSave(saveId):
    characterData = CacheContorl.characterData
    gameTime = CacheContorl.gameTime
    gameVerson = GameConfig.verson
    scaneData = CacheContorl.sceneData
    mapData = CacheContorl.mapData
    npcTemData = CacheContorl.npcTemData
    randomNpcList = CacheContorl.randomNpcList
    data = {"characterData":characterData,"gameTime":gameTime,"gameVerson":gameVerson,"sceneData":scaneData,"mapData":mapData,"npcTemData":npcTemData,"randomNpcList":randomNpcList}
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
        EraPrint.p(TextLoading.getTextData(TextLoading.errorPath,'notSaveError'))
    return data

# 确认存档读取
def inputLoadSave(saveId):
    saveData = loadSave(saveId)
    CacheContorl.characterData = saveData['characterData']
    CacheContorl.characterData['characterId'] = '0'
    CacheContorl.gameTime = saveData['gameTime']
    CacheContorl.sceneData = saveData['sceneData']
    CacheContorl.mapData = saveData['mapData']
    CacheContorl.npcTemData = saveData['npcTemData']
    CacheContorl.randomNpcList = saveData['randomNpcList']
    CharacterHandle.initCharacterPosition()

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
        errorText = TextLoading.getTextData(TextLoading.errorPath,'notSaveError')
        EraPrint.pl(errorText)
