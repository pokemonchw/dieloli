DieLoli
====
死亡萝莉
====

介绍
----
这是一个[ERALIKE游戏](http://www.emuera.net/) \
在游戏中，你将与多达2800位，由算法模拟整个人生成长过程，随机生成的游戏角色一起，在校园里生活 \
你将在游戏中体验不一样的人生，这里有魔法少女，有触手怪，有吸血鬼，也有巫女，妖怪，幽灵，和帅气的魔法师 \
当然，这些都需要你自己去进行探索，至少，你也可以将它当成一个普通的校园恋爱模拟游戏玩 \
游戏中基于元胞算法进行即时演算，所有事件的脉络和走向都不是预设好的 \
每个角色都有各自的经历和成长，不断的学习和生活 \
在这里，没有人是特殊的，也没有人可以掌控世界，玩家也不行

著作权信息
----
此项目基于Python独立进行开发 \
早期版本曾使用[pyera](https://github.com/qsjl11/pyera/) 进行构架，现已全部废弃 \
基于[cc by nc sa](http://creativecommons.org/licenses/by-nc-sa/2.0/) 协议，开发者允许任何人基于此项目做除商业行为外的任何事，同时允许任何人对本项目进行除协议外的任何改动，仅需注明原作者，以及以相同方式进行传播(指同样使用cc by-nc-sa协议) \
请勿将其用作商业用途 \
想要参与开发请与开发者联系，联系方式在本md底部

Repo说明
----
日常开发备份在instability分支中进行,pr也请提交至此分支 \
master分支在开发完成前作为设计展示分支使用 \
代码风格化通常使用black自动完成，行宽为108

请求
----
帮助中国的可怜萝莉控!

配置要求
----
GPU: \
本游戏几乎没有任何显卡要求，如你所见，它是个纯文字游戏 \
CPU: \
在默认配置(2800个游戏角色)下，2015年9月发布的i5 6200U和2019年9月发布的i7 10710U仅差了23%，几乎可以忽略不计(由于游戏性设计的原因，本游戏为祖传单核游戏引擎) \
Memory: \
在默认配置(2800个游戏角色)下，游戏占用总内存不超过250MB \
系统: \
本游戏兼容archlinux/steamos/chromeos/ubuntu/debian/aoscos等绝大部分支持gui的linux系操作系统，同时也可以在macos和windows7及以上操作系统中运行

依赖
----
python3.9.0

建议通过::

    pip install -r requirements.txt

进行安装

字体
----
本游戏界面设计依赖Sarasa Mono SC字体，若系统未安装此字体将会fallback到系统默认字体，不能保证能否达到设计效果 \
字体相关配置可以通过data/FontConfig.json更改 \
本游戏不提供Sarasa Mono SC相关字体文件 \
请自行下载并安装:[Sarasa Mono SC](https://github.com/be5invis/Sarasa-Gothic)

本地化
----
本项目使用gettext进行本地化设置 \
请于 data/po 目录下创建对应语言目录 \
切换语言请编辑config.ini中的language项 \
协作翻译方案待定

警告
----
在游戏设计中，有一定的色情，暴力，宗教，等内容，在进行游戏前请确认自己的三观足够健全到不会随意被动摇和影响

联系方式
----
e-mail:admin@byayoi.org \
twitter:@nekoharuyayuzu

请我喝奶茶QwQ
----
请前往:[爱发电](https://afdian.net/@byayoi)订阅
