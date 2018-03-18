import core.EraPrint as eprint
import core.GameConfig as config
import script.Panel.SaveHandleFramePanel as savehandleframepanel

# 绘制保存存档页面流程
def establishSave_func():
    maxSaveValue = int(config.max_save)
    pageSaveValue = int(config.save_page)
    lastSavePageValue = 0
    if maxSaveValue % pageSaveValue != 0:
        showSaveValue = int(maxSaveValue / pageSaveValue)
        lastSavePageValue = maxSaveValue % pageSaveValue
    else:
        showSaveValue = maxSaveValue / pageSaveValue
    savehandleframepanel.seeSaveListPanel(showSaveValue,lastSavePageValue)
    pass

# 读取存档界面
def loadSave_func():
    eprint.p('\n')
    eprint.pline()
