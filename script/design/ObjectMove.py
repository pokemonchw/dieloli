import design.MapHandle as maphandle
import core.CacheContorl as cache
import flow.InScene as inscene

# 主角移动
def playerMove(targetScene):
    playerPosition = cache.playObject['object']['0']['position']
    if playerPosition == targetScene:
        inscene.getInScene_func()
    else:

        pass
    pass