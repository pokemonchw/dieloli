import os
from tkinter import END
from PIL.ImageTk import PhotoImage
from Script.Core import game_path_config, main_frame, game_type, cache_control

textbox = main_frame.textbox
image_data = {}
image_text_data = {}
image_lock = 0
cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """


def get_image_data(image_name: str, image_path: str = "") -> PhotoImage:
    """
    按路径读取图片数据并创建PhotoImage对象
    Keyword arguments:
    image_name -- 图片名字
    image_path -- 图片路径 (default '')
    """
    if image_path == "":
        image_path = os.path.join("image", image_name + ".png")
    else:
        image_path = os.path.join("image", image_path, image_name + ".png")
    cache.image_id += 1
    return PhotoImage(file=image_path)


def print_image(image_name: str, image_path: str = ""):
    """
    绘制图片的内部实现，按图片id将图片加入绘制队列
    Keyword arguments:
    image_name -- 图片名字
    image_path -- 图片路径 (default '')
    """
    image_data[str(cache.image_id)] = get_image_data(image_name, image_path)
    textbox.image_create(END, image=image_data[str(cache.image_id)])
