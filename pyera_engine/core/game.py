# -*- coding: UTF-8 -*-
import core.data
import core.flow
import core.pycfg
import core.pyio as pyio
import core.Event as event

# 字符串定义###########################################################
NO_EVENT_FUNC='no_event_func'

# 系统函数#############################################################
# 初始化函数
_main_flow = None


def init(main_flow):
    global def_style
    pyio.clear_screen()
    pyio.clearorder()
    core.flow.cmd_clear()
    # 载入数据库数据
    core.data.init()
    # 事件载入
    event.load_event_file()
    # 设置背景颜色
    core.data._get_savefilename_path('')
    pyio.set_background(core.data.gamedata()['core_cfg']['background_color'])
    foreground_c = core.data.gamedata()['core_cfg']['font_color']
    background_c = core.data.gamedata()['core_cfg']['background_color']
    onbutton_color = core.data.gamedata()['core_cfg']['onbutton_color']
    font = core.data.gamedata()['core_cfg']['font']
    fontsize = core.data.gamedata()['core_cfg']['font_size']
    pyio.init_style(foreground_c, background_c, onbutton_color, font, fontsize)
    pyio.style_def('warning', foreground='red', underline=True)
    pyio.style_def('special', foreground='yellow')
    pyio.style_def('grey',foreground='grey')
    def_style = pyio.style_def
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


def run(main_func):
    """运行函数"""
    def _init():
        init(main_func)
    core.pyio.run(_init)


def console_log(string):
    """向控制台输入信息"""
    print('game log:')
    print(string + '\n')


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
askfor_int = core.flow.askfor_int

# 设置尾命令处理函数
set_deal_cmd_func=core.flow.set_tail_deal_cmd_func

# 设置尾命令处理函数装饰器
set_deal_cmd_func_deco=core.flow.deco_set_tail_deal_cmd_func

# 返回主数据集合
data = core.data.gamedata()

# 获得存档目录
savedir = core.data._get_savefilename_path('')[:-6]

# 保存数据集合到文件, 也可将可以game.data序列化保存到某个文件中
save = core.data.save

# 从文件中加载数据集合, selfdata为True时，只返回反序列化之后的数据，不会将数据加载到gamedata
load = core.data.load
