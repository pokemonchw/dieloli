import os,pickle,shutil
from script.Core import EraPrint,TextLoading,CacheContorl,GameConfig,GamePathConfig
from script.Design import CharacterHandle

gamepath = GamePathConfig.gamepath

# 获取存档所在路径
def getSaveDirPath(saveId):
    saveId = str(saveId)
    savepath = os.path.join(gamepath,'save')
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    return os.path.join(savepath,saveId)

# 判断存档是否存在
def judgeSaveFileExist(saveId):
    savePath = getSaveDirPath(saveId)
    if not os.path.exists(savePath):
        return "0"
    return "1"

# 存入存档数据
def establishSave(saveId):
    characterData = CacheContorl.characterData
    gameTime = CacheContorl.gameTime
    scaneData = CacheContorl.sceneData
    mapData = CacheContorl.mapData
    npcTemData = CacheContorl.npcTemData
    randomNpcList = CacheContorl.randomNpcList
    gameVerson = GameConfig.verson
    saveVerson = {
        "gameVerson":gameVerson,
        "gameTime":gameTime,
        "characterName":characterData['character']['0']['Name']
    }
    data = {"1":characterData,"2":gameTime,"0":saveVerson,"3":scaneData,"4":mapData,"5":npcTemData,"6":randomNpcList}
    for dataId in data:
        writeSaveData(saveId,dataId,data[dataId])

# 载入存档信息头
def loadSaveInfoHead(saveId):
    savePath = getSaveDirPath(saveId)
    filePath = os.path.join(savePath,'0')
    with open(filePath,'rb') as f:
        return pickle.load(f)

# 写入存档数据
def writeSaveData(saveId,dataId,writeData):
    savePath = getSaveDirPath(saveId)
    filePath = os.path.join(savePath,dataId)
    if judgeSaveFileExist(saveId) == '0':
        os.makedirs(savePath)
    with open(filePath,'wb') as f:
        pickle.dump(writeData,f)

# 读取存档数据
def loadSave(saveId):
    savePath = getSaveDirPath(saveId)
    data = {}
    fileList = ['1','2','3','4','5','6']
    for fileName in fileList:
        filePath = os.path.join(savePath,fileName)
        with open(filePath, 'rb') as f:
            data[fileName]=pickle.load(f)
    return data

# 确认存档读取
def inputLoadSave(saveId):
    saveData = loadSave(saveId)
    CacheContorl.characterData = saveData['1']
    CacheContorl.characterData['characterId'] = '0'
    CacheContorl.gameTime = saveData['2']
    CacheContorl.sceneData = saveData['3']
    CacheContorl.mapData = saveData['4']
    CacheContorl.npcTemData = saveData['5']
    CacheContorl.randomNpcList = saveData['6']
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
    savePath = getSaveDirPath(saveId)
    if os.path.isdir(savePath):
        shutil.rmtree(savePath)
    else:
        errorText = TextLoading.getTextData(TextLoading.errorPath,'notSaveError')
        EraPrint.pl(errorText)
