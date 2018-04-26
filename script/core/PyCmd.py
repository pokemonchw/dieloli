import core.flow as flow
import core.winframe as winframe
import core.Dictionaries as dictionaries
import core.CacheContorl as cache
import core.GameConfig as config

#清除命令
clear_default_flow=flow.clear_default_flow
#绑定或重新绑定一个命令
bind_cmd = flow.bind_cmd

#输出命令
def pcmd(cmd_str, cmd_id, cmd_func=flow.null_func, arg=(), kw={}, normal_style='standard', on_style='onbutton'):
    cmd_str = dictionaries.handleText(cmd_str)
    cmd_id = dictionaries.handleText(str(cmd_id))
    cache.textWait = float(config.text_wait)
    global last_char
    if len(cmd_str) > 0:
        last_char = cmd_str[-1:]
    flow.print_cmd(cmd_str, cmd_id, cmd_func, arg, kw, normal_style, on_style)

#获得一个没有用过的命令编号
unused_cmd_num=500
def get_unused_cmd_num():
    global unused_cmd_num
    unused_cmd_num +=1
    return unused_cmd_num

#清除命令，没有参数则清除所有命令
def clr_cmd(*number, clr_default_flow=True):
    if clr_default_flow==True:
        clear_default_flow()
    if number:
        flow.cmd_clear(number)
    else:
        global  unused_cmd_num
        unused_cmd_num = 500
        flow.cmd_clear()

def focusCmd():
    winframe.inputbox.focus_force()