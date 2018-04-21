import core.CacheContorl as cache
import core.TextLoading as textload
import core.EraPrint as eprint
import script.GameTime as gametime

# 用于查看当前场景的面板
def seeScenePanel():
    sceneData = cache.sceneData
    titleText = textload.getTextData(textload.stageWordId,'75')
    eprint.plt(titleText)
    timeText = gametime.getDateText()
    eprint.p(timeText)
    eprint.plittleline()

    pass