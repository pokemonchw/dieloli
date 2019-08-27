# -*- coding: UTF-8 -*-
from script.Core import GameData,FlowHandle,IoInit,KeyListionEvent,GameConfig,CacheContorl

# 字符串定义###########################################################
NO_EVENT_FUNC='no_event_func'

# 系统函数#############################################################
# 初始化函数
_main_flow = None

def init(main_flow):
    '''
    游戏流程初始化
    Keyword argument:
    main_flow -- 游戏主流程
    '''
    global def_style
    IoInit.clear_screen()
    IoInit.clearorder()
    FlowHandle.cmd_clear()
    # 载入按键监听
    KeyListionEvent.onWFrameListion()
    # 设置背景颜色
    IoInit.set_background(GameData.gamedata()['core_cfg']['background_color'])
    # 初始化字体
    IoInit.init_style()
    # 初始化地图数据
    CacheContorl.mapData = GameData._gamedata[GameConfig.language]['map']
    CacheContorl.sceneData = GameData.sceneData
    CacheContorl.mapData = GameData.mapData
    FlowHandle.reset_func = reset
    global _main_flow
    _main_flow = main_flow

    _have_run=False
    def run_main_flow():
        nonlocal _have_run
        while True:
            if _have_run==False:
                main_flow()
                _have_run=True
            askfor_order()
            FlowHandle.call_default_flow()
            if FlowHandle.exit_flag==True:
                break

    run_main_flow()

def run(main_func):
    '''
    执行游戏主流程
    Keyword arguments:
    main_func -- 游戏主流程
    '''
    def _init():
        init(main_func)
    IoInit.run(_init)

def console_log(string):
    '''
    向后台打印日志
    Keyword arguments:
    string -- 游戏日志信息
    '''
    print('game log:')
    print(string + '\n')

def reset():
    '''
    重启游戏
    '''
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
