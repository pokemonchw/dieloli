# -*- coding: UTF-8 -*-

import core.game as game


def yes_or_no():
    game.pcmd('[0] 是  ', 0, None)
    game.pcmd('[1] 否  ', 1, None)
    while True:
        ans = game.askfor_int()
        if ans == 0:
            return True
        if ans == 1:
            return False


def list_cmd(print_list, func_list, first_cmd_num=None, spilt_mark=' / ', default_position=None,
             default_style='standard'):
    if len(print_list) != len(func_list):
        game.pwarn("list_choice调用,print_list和func_list不匹配")
    for n in range(len(print_list)):
        if first_cmd_num == None:
            cmd_num = game.get_unused_cmd_num()
        else:
            cmd_num = first_cmd_num + n
        cmd_str = '[' + str(cmd_num) + '] ' + str(print_list[n]) + spilt_mark
        if default_position == n:
            game.pcmd(cmd_str, cmd_num, func_list[n], normal_style=default_style)
        else:
            game.pcmd(cmd_str, cmd_num, func_list[n])


def list_nums(num_list, set_func, current_value, first_cmd_num=None, split_mark=' / '):
    def create_func(n):
        def _func():
            set_func(n)

        return _func

    func_list = []
    for nn in num_list:
        func_list.append(create_func(nn))
    list_cmd(num_list, func_list, first_cmd_num, split_mark, default_position=num_list.index(current_value),
             default_style='special')


def get_id():
    if not 'last_id' in game.data:
        game.data['last_id'] = 100
    game.data['last_id'] += 1
    return game.data['last_id']


def value_bar(vnow, vmax, length=30):
    if length < 3:
        length = 30
    show_sample = int(vnow / vmax * length)
    if show_sample > length:
        show_sample = length
    if show_sample < 0:
        show_sample = 0
    string = '[{0}{1}] ({2}/{3})'.format('*' * show_sample, '-' * (length - show_sample), vnow, vmax)
    return string


def get_e_by_ID(id_num, list):
    for p in list:
        if p['ID'] == id_num:
            return p
    return None


class IterExceptNoneInList():
    def __init__(self, list):
        self.l = iter(list)

    def __iter__(self):
        return self

    def __next__(self):
        nextpeople = None
        try:
            while nextpeople == None:
                nextpeople = next(self.l)
            return nextpeople
        except StopIteration:
            raise StopIteration


import os

def load_func(return_func, next_func):
    game.clr_cmd()
    game.pl('读取游戏：' + game.savedir)
    game.pline()

    def loadhere(load_file_name):
        game.pl('load: ' + load_file_name)
        game.load(load_file_name)
        next_func()

    def loadnodata(load_file_name):
        game.pl(load_file_name + ": 没有数据")


    for i in range(0, 11):
        load_file_name = 'save' + str(i)
        load_file_path = game.savedir + '\\' + load_file_name + '.save'

        # 此处修改显示信息
        if os.path.exists(load_file_path):
            file_str = '[{:0>2}]'.format(i) + "  " + load_file_path
            game.pcmd(file_str, i, loadhere, arg=(load_file_name,))
        else:
            file_str = '[{:0>2}]'.format(i) + "  ----"
            game.pcmd(file_str, i, loadnodata, arg=(load_file_name,))
        game.pl()
    game.pl()
    game.pcmd('[99] 返回', 99, return_func)



def save_func(return_func):
    game.clr_cmd()
    game.pline()
    game.pl('游戏存储目录：' + game.savedir)
    game.pl()

    def savehere(save_file_name):
        game.pl('save: ' + save_file_path)
        game.save(save_file_name)
        save_func(return_func)

    for i in range(0, 11):
        save_file_name = 'save' + str(i)
        save_file_path = game.savedir + '\\' + save_file_name + '.save'

        # 此处修改显示信息
        if os.path.exists(save_file_path):
            file_str = '[{:0>2}]'.format(i) + "  " + save_file_path
        else:
            file_str = '[{:0>2}]'.format(i) + "  ----"
        game.pcmd(file_str, i, savehere, arg=(save_file_name,))
        game.pl()
    game.pl()
    game.pcmd('[99] 返回', 99, return_func)


