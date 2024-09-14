import os
from PIL.ImageTk import PhotoImage
from PIL import Image
from Script.Core import main_frame, game_type, cache_control, game_path_config


image_data = {}
image_text_data = {}
image_lock = 0
cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """
image_dir_path = game_path_config.resource_path(os.path.join("image"))
for image_file_path_id in os.listdir(image_dir_path):
    image_file_path = os.path.join(image_dir_path, image_file_path_id)
    image_file_name = image_file_path_id.rstrip(".png")
    old_image = PhotoImage(file=image_file_path)
    old_height = old_image.height()
    old_weight = old_image.width()
    font_width_scaling = main_frame.now_char_width / 12
    font_height_scaling = main_frame.now_char_height / 24
    now_height = int(old_height * font_height_scaling)
    now_weight = int(old_weight * font_width_scaling)
    new_image = Image.open(image_file_path).resize((now_weight, now_height))
    image_data[image_file_name] = PhotoImage(new_image)
