# -*- coding: UTF-8 -*-
import time,os
from script.Core import CacheContorl,TextLoading,TextHandle,GameConfig,IoInit

def null_func():
    return

# 管理flow
default_flow=null_func

def set_default_flow(func, arg=(), kw={}):
    global  default_flow
    if not isinstance(arg, tuple):
        arg = (arg,)
    if func==null_func:
        default_flow = null_func
        return

    def run_func():
        func(*arg, **kw)
    default_flow = run_func

def call_default_flow():
    default_flow()

def clear_default_flow():
    global default_flow, null_func
    set_default_flow(null_func)

cmd_map = CacheContorl.cmd_map

def default_tail_deal_cmd_func(order):
    return


tail_deal_cmd_func = default_tail_deal_cmd_func


def set_tail_deal_cmd_func(func):
    global tail_deal_cmd_func
    tail_deal_cmd_func = func

def deco_set_tail_deal_cmd_func(func):
    set_tail_deal_cmd_func(func)
    return func

def bind_cmd(cmd_number, cmd_func, arg=(), kw={}):
    if not isinstance(arg, tuple):
        arg = (arg,)
    if cmd_func==null_func:
        cmd_map[cmd_number] = null_func
        return

    def run_func():
        cmd_func(*arg, **kw)
    cmd_map[cmd_number] = run_func

def print_cmd(cmd_str, cmd_number, cmd_func=null_func, arg=(), kw={}, normal_style='standard', on_style='onbutton'):
    '''arg is tuple contain args which cmd_func could be used'''
    bind_cmd(cmd_number, cmd_func, arg, kw)
    IoInit.io_print_cmd(cmd_str, cmd_number, normal_style, on_style)
    return cmd_str


def cmd_clear(*number):
    set_tail_deal_cmd_func(default_tail_deal_cmd_func)
    if number:
        for num in number:
            cmd_map.pop(num)
            IoInit.io_clear_cmd(num)
    else:
        cmd_map.clear()
        IoInit.io_clear_cmd()


def _cmd_deal(order_number):
    cmd_map[int(order_number)]()


def _cmd_valid(order_number):
    re=(order_number in cmd_map.keys()) and (cmd_map[int(order_number)] != null_func)
    return re


__skip_flag__ = False
reset_func = None
exit_flag =False

# 处理输入
def order_deal(flag='order', print_order=True):
    global __skip_flag__
    __skip_flag__ = False
    while True:
        time.sleep(0.01)
        while not IoInit._order_queue.empty():
            order = IoInit.getorder()
            if CacheContorl.flowContorl['quitGame']:
                os._exit(0)
                return
            if CacheContorl.flowContorl['restartGame'] == 1:
                CacheContorl.flowContorl['restartGame'] = 0
                reset_func()
                return
            if print_order == True and order != '':
                IoInit.print('\n' + order + '\n')

            if flag == 'str':
                return order

            if flag == 'console':
                # TODO add_console_method
                exec(order)

            if flag == 'order' and order.isdigit():
                if _cmd_valid(int(order)):
                    _cmd_deal(int(order))
                    return
                else:
                    global tail_deal_cmd_func
                    tail_deal_cmd_func(int(order))
                    return


def askfor_str(donot_return_null_str=True, print_order=False):
    while True:
        order = order_deal('str', print_order)
        if donot_return_null_str == True and order != '':
            return order
        elif donot_return_null_str == False:
            return order


def askfor_All(list,print_order=False):
    while True:
        order = order_deal('str', print_order)
        if order in list:
            IoInit.print(order + '\n')
            return order
        elif order == '':
            continue
        else:
            IoInit.print(order + '\n')
            IoInit.print(TextLoading.getTextData(TextLoading.errorId, 'noInputListError') + '\n')
            continue

def askfor_Int(list,print_order=False):
    while True:
        order = order_deal('str', print_order)
        order = TextHandle.fullToHalfText(order)
        if order in list:
            IoInit.print(order + '\n')
            return order
        elif order == '':
            continue
        else:
            IoInit.print(order + '\n')
            IoInit.print(TextLoading.getTextData(TextLoading.errorId, 'noInputListError') + '\n')
            continue

def askfor_wait():
    global __skip_flag__
    while __skip_flag__ == False:
        re = askfor_str(donot_return_null_str=False)
        if re == '':
            break

def initCache():
    CacheContorl.flowContorl = {'restartGame': 0, 'quitGame': 0}
    CacheContorl.wframeMouse = {'wFrameUp': 2, 'mouseRight': 0, 'mouseLeaveCmd': 1, 'wFrameLinesUp': 2, 'wFrameLineState': 2,
                   'wFrameRePrint': 0}
    CacheContorl.cmd_map = {}
    CacheContorl.playObject = {'objectId': '', 'object': {}}
    CacheContorl.featuresList = {'Age': "", "Chastity": "", 'Disposition': "", 'Courage': "", 'SelfConfidence': "", 'Friends': "",
                    'Figure': "", 'Sex': "", 'AnimalInternal': "", 'AnimalExternal': "", 'Charm': ""}
    CacheContorl.temporaryObject = {}
    CacheContorl.inputCache = ['']
    CacheContorl.inputPosition = {'position': 0}
    CacheContorl.outputTextStyle = 'standard'
    CacheContorl.textStylePosition = {'position': 0}
    CacheContorl.textStyleCache = ['standard']
    CacheContorl.textOneByOneRichCache = {'textList': [], 'styleList': []}
    CacheContorl.gameTime = {"year":0,"month":0,"day":0,"hour":0,"minute":0}
    CacheContorl.cmdData = {}
    CacheContorl.imageid = 0
    CacheContorl.panelState = {
        "PlayerMainAttrPanel":"1","PlayerEquipmentPanel":"1",
        "PlayerItemPanel":"1","PlayerExperiencePanel":"1",
        "PlayerLevelPanel":"1","PlayerFeaturesPanel":"1",
        "PlayerEngravingPanel":"1","AttrShowHandlePanel":"0",
        "SeeSaveListPanel":"0","SeePlayerListPanel":"0"
    }
    CacheContorl.maxSavePage = GameConfig.save_page
    CacheContorl.textWait = float(GameConfig.text_wait)
    CacheContorl.temObjectDefault = getTemObjectDefault()
    CacheContorl.temporaryObjectBak = CacheContorl.temObjectDefault.copy()

def getTemObjectDefault():
    script = __import__('script.Design.MapHandle')
    temObject = TextLoading.getTextData(TextLoading.roleId, 'Default')
    temPositionDirList = temObject['Position']
    temPosition = script.Design.MapHandle.getSceneIdForDirList(temPositionDirList)
    temObject['Position'] = temPosition
    return temObject