# -*- coding: UTF-8 -*-
from script.Core import GameData,FlowHandle,IoInit,Event,KeyListionEvent
from script.Design import MapHandle

# 字符串定义###########################################################
NO_EVENT_FUNC='no_event_func'

# 系统函数#############################################################
# 初始化函数
_main_flow = None

# 游戏初始化
def init(main_flow):
    global def_style
    IoInit.clear_screen()
    IoInit.clearorder()
    FlowHandle.cmd_clear()
    # 载入数据库数据
    GameData.init()
    # 事件载入
    Event.load_event_file()
    # 载入按键监听
    KeyListionEvent.onWFrameListion()
    # 设置背景颜色
    IoInit.set_background(GameData.gamedata()['core_cfg']['background_color'])
    # 初始化字体
    IoInit.init_style()
    # 初始化地图数据
    MapHandle.initSceneData()
    MapHandle.initMapData()

    FlowHandle.reset_func = reset

    global _main_flow
    _main_flow = main_flow

    _have_run=False
    def run_main_flow():
        nonlocal  _have_run
        while True:
            if _have_run==False:
                main_flow()
                _have_run=True
            askfor_order()
            FlowHandle.call_default_flow()
            if FlowHandle.exit_flag==True:
                break

    run_main_flow()

# 运行函数
def run(main_func):
    def _init():
        init(main_func)
    IoInit.run(_init)

# 向控制台输入信息
def console_log(string):
    print('game log:')
    print(string + '\n')

# 重启游戏
def reset():
    global _main_flow
    IoInit.io_clear_cmd()
    IoInit.clear_screen()
    IoInit.clearorder()
    init(_main_flow)

# 请求输入命令
askfor_order = FlowHandle.order_deal

# 请求输入一个字符串
askfor_str = FlowHandle.askfor_str

# 请求输入一个数字
askfor_Int = FlowHandle.askfor_Int
askfor_All = FlowHandle.askfor_All

# 设置尾命令处理函数
set_deal_cmd_func = FlowHandle.set_tail_deal_cmd_func

# 设置尾命令处理函数装饰器
set_deal_cmd_func_deco = FlowHandle.deco_set_tail_deal_cmd_func

# 返回主数据集合
data = GameData.gamedata()
