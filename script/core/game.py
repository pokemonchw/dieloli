# -*- coding: UTF-8 -*-
import core.data
import core.flow
import core.pycfg
import core.pyio as pyio
import core.Event as event
import core.KeyListionEvent as keylistion
import design.MapHandle as maphandle

# 字符串定义###########################################################
NO_EVENT_FUNC='no_event_func'

# 系统函数#############################################################
# 初始化函数
_main_flow = None

# 游戏初始化
def init(main_flow):
    global def_style
    pyio.clear_screen()
    pyio.clearorder()
    core.flow.cmd_clear()
    # 载入数据库数据
    core.data.init()
    # 事件载入
    event.load_event_file()
    # 载入按键监听
    keylistion.onWFrameListion()
    # 设置背景颜色
    pyio.set_background(core.data.gamedata()['core_cfg']['background_color'])
    # 初始化字体
    pyio.init_style()
    # 初始化地图数据
    maphandle.initSceneData()
    maphandle.initMapData()

    core.flow.reset_func = reset

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
            core.flow.call_default_flow()
            if core.flow.exit_flag==True:
                break

    run_main_flow()

# 运行函数
def run(main_func):
    def _init():
        init(main_func)
    core.pyio.run(_init)

# 向控制台输入信息
def console_log(string):
    print('game log:')
    print(string + '\n')

# 重启游戏
def reset():
    global _main_flow
    pyio.io_clear_cmd()
    pyio.clear_screen()
    pyio.clearorder()
    init(_main_flow)

# 请求输入命令
askfor_order = core.flow.order_deal

# 请求输入一个字符串
askfor_str = core.flow.askfor_str

# 请求输入一个数字
askfor_Int = core.flow.askfor_Int
askfor_All = core.flow.askfor_All

# 设置尾命令处理函数
set_deal_cmd_func=core.flow.set_tail_deal_cmd_func

# 设置尾命令处理函数装饰器
set_deal_cmd_func_deco=core.flow.deco_set_tail_deal_cmd_func

# 返回主数据集合
data = core.data.gamedata()