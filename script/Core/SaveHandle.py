import os
import pickle
import shutil
from script.Core import EraPrint,TextLoading,CacheContorl,GameConfig,GamePathConfig
from script.Design import CharacterHandle

gamepath = GamePathConfig.gamepath

def getSaveDirPath(saveId:str) -> str:
    '''
    按存档id获取存档所在系统路径
    Keyword arguments:
    saveId -- 存档id
    '''
    savepath = os.path.join(gamepath,'save')
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    return os.path.join(savepath,saveId)

def judgeSaveFileExist(saveId:str) -> bool:
    '''
    判断存档id对应的存档是否存在
    Keyword arguments:
    saveId -- 存档id
    '''
    savePath = getSaveDirPath(saveId)
    if not os.path.exists(savePath):
        return False
    return True

def establishSave(saveId:str):
    '''
    将游戏数据存入指定id的存档内
    Keyword arguments:
    saveId -- 存档id
    '''
    characterData = CacheContorl.characterData
    gameTime = CacheContorl.gameTime
    scaneData = CacheContorl.sceneData
    mapData = CacheContorl.mapData
    npcTemData = CacheContorl.npcTemData
    randomNpcList = CacheContorl.randomNpcList
    gameVerson = GameConfig.verson
    occupationCharacterData = CacheContorl.occupationCharacterData
    totalBodyFatByage = CacheContorl.TotalBodyFatByage
    averageBodyFatbyage = CacheContorl.AverageBodyFatByage
    totalNumberOfPeopleOfAllAges = CacheContorl.TotalNumberOfPeopleOfAllAges
    totalHeightByage = CacheContorl.TotalHeightByage
    averageHeightByage = CacheContorl.AverageHeightByage
    saveVerson = {
        "gameVerson":gameVerson,
        "gameTime":gameTime,
        "characterName":characterData['character']['0']['Name']
    }
    data = {"1":characterData,"2":gameTime,"0":saveVerson,"3":scaneData,"4":mapData,"5":npcTemData,"6":randomNpcList,"7":occupationCharacterData,"8":totalBodyFatByage,"9":averageBodyFatbyage,"10":totalNumberOfPeopleOfAllAges,"11":totalHeightByage,"12":averageHeightByage}
    for dataId in data:
        writeSaveData(saveId,dataId,data[dataId])

def loadSaveInfoHead(saveId:str) -> dict:
    '''
    获取存档的头部信息
    Keyword arguments:
    saveId -- 存档id
    '''
    savePath = getSaveDirPath(saveId)
    filePath = os.path.join(savePath,'0')
    with open(filePath,'rb') as f:
        return pickle.load(f)

def writeSaveData(saveId:str,dataId:str,writeData:dict):
    '''
    将存档数据写入文件
    Keyword arguments:
    saveId -- 存档id
    dataId -- 要写入的数据在存档下的文件id
    writeData -- 要写入的数据
    '''
    savePath = getSaveDirPath(saveId)
    filePath = os.path.join(savePath,dataId)
    if judgeSaveFileExist(saveId) == False:
        os.makedirs(savePath)
    with open(filePath,'wb') as f:
        pickle.dump(writeData,f)

def loadSave(saveId:str) -> dict:
    '''
    按存档id读取存档数据
    Keyword arguments:
    saveId -- 存档id
    '''
    savePath = getSaveDirPath(saveId)
    data = {}
    fileList = ['1','2','3','4','5','6','7','8','9','10','11','12']
    for fileName in fileList:
        filePath = os.path.join(savePath,fileName)
        with open(filePath, 'rb') as f:
            data[fileName]=pickle.load(f)
    return data

def inputLoadSave(saveId:str):
    '''
    载入存档存档id对应数据，覆盖当前游戏内存
    Keyword arguments:
    saveId -- 存档id
    '''
    saveData = loadSave(saveId)
    CacheContorl.characterData = saveData['1']
    CacheContorl.characterData['characterId'] = '0'
    CacheContorl.gameTime = saveData['2']
    CacheContorl.sceneData = saveData['3']
    CacheContorl.mapData = saveData['4']
    CacheContorl.npcTemData = saveData['5']
    CacheContorl.randomNpcList = saveData['6']
    CacheContorl.occupationCharacterData = saveData['7']
    CacheContorl.TotalBodyFatByage = saveData['8']
    CacheContorl.AverageBodyFatByage = saveData['9']
    CacheContorl.TotalNumberOfPeopleOfAllAges = saveData['10']
    CacheContorl.TotalHeightByage = saveData['11']
    CacheContorl.AverageHeightByage = saveData['12']
    CharacterHandle.initCharacterPosition()

def getSavePageSaveId(pageSaveValue:int,inputId:int) -> int:
    '''
    按存档页计算，当前页面输入数值对应存档id
    Keyword arguments:
    pageSaveValue -- 存档页Id
    inputId -- 当前输入数值
    '''
    savePanelPage = int(CacheContorl.panelState['SeeSaveListPanel']) + 1
    startSaveId = pageSaveValue * (savePanelPage - 1)
    saveId = startSaveId + inputId
    return int(saveId)

def removeSave(saveId:str):
    '''
    删除存档id对应存档
    Keyword arguments:
    saveId -- 存档id
    '''
    savePath = getSaveDirPath(saveId)
    if os.path.isdir(savePath):
        shutil.rmtree(savePath)
    else:
        errorText = TextLoading.getTextData(TextLoading.errorPath,'notSaveError')
        EraPrint.pl(errorText)
