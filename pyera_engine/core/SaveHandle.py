import os
try:
    import cPickle as pickle
except ImportError:
    import pickle
import core.EraPrint as eprint
import core.TextLoading as textload

# 获取存档所在路径
def getSavefilePath(filename):
    global gamepath
    savepath = os.path.join(gamepath,'save')
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    filepath = os.path.join(savepath,filename + '.save')
    return filepath

# 判断存档是否存在
def judgeSaveFileExist(saveName,saveId):
    global gamepath
    savePath = os.path.join(gamepath,'save')
    if not os.path.exists(savePath):
        return "0"
    else:
        return "1"

# 存入存档数据
def establishSave(filename, data=None):
    if data == None:
        data = _gamedata
    filepath = getSavefilePath(filename)
    with open(filepath, 'wb') as f:
        pickle.dump(data,f)

# 读取存档数据
def loadSave(filename, selfdata=False):
    filepath = getSavefilePath(filename)
    data = {}
    try:
        with open(filepath, 'rb') as f:
            data=pickle.load(f)
    except FileNotFoundError:
        eprint.p(textload.getTextData(textload.errorId,'notSaveError'))
    if selfdata == False:
        global _gamedata
        _gamedata.update(data)
    return data