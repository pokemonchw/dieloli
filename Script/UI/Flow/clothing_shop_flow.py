from Script.Design import handle_panel, constant
from Script.UI.Panel import clothing_shop_panel
from Script.Config import normal_config

width = normal_config.config_normal.text_width
""" 屏幕宽度 """


@handle_panel.add_panel(constant.Panel.CLOTHING_SHOP)
def clothing_shop_flow():
    """服装商店面板"""
    now_panel = clothing_shop_panel.ClothingShopPanel(width)
    now_panel.draw()
