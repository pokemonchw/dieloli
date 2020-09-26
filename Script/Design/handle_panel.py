from functools import wraps
from Script.Core import cache_contorl

def add_panel(panel:str) -> callable:
    """
    添加面板
    Keyword arguments:
    panel -- 面板id
    Return arguments:
    callable -- 面板绘制对象
    """
    def decoraror(func):
        @wraps(func)
        def return_wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        cache_contorl.panel_data[panel] = return_wrapper
        return return_wrapper

    return decoraror
