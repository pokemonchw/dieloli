from script.Core import CacheContorl,TextLoading,EraPrint,PyCmd
from script.Design import CmdButtonQueue

def seeInstructHeadPanel() -> list:
    '''
    绘制指令面板的头部过滤器面板
    Return arguments:
    list -- 绘制的按钮列表
    '''
    EraPrint.plt(TextLoading.getTextData(TextLoading.stageWordPath,'146'))
    instructData = TextLoading.getTextData(TextLoading.cmdPath,CmdButtonQueue.instructheadpanel)
    if CacheContorl.instructFilter == {}:
        CacheContorl.instructFilter = {instruct:0 for instruct in instructData}
        CacheContorl.instructFilter['Dialogue'] = 1
    styleData = {instructData[instruct]:"selectmenu" for instruct in instructData if CacheContorl.instructFilter[instruct] == 0}
    onStyleData = {instructData[instruct]:"onselectmenu" for instruct in instructData if CacheContorl.instructFilter[instruct] == 0}
    EraPrint.p(TextLoading.getTextData(TextLoading.stageWordPath,'147'))
    return CmdButtonQueue.optionstr(None,len(instructData),'center',False,False,list(instructData.values()),'',list(instructData.keys()),styleData,onStyleData)
