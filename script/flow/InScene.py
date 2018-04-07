import core.CacheContorl as cache
import script.Panel.InScenePanel as inscenepanel

# 用于查看当前场景的流程
def seeScene_func():
    mapData = cache.mapData
    inscenepanel.seeScenePanel()
    pass