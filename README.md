DieLoli
====
死亡萝莉
====

介绍
----
这是一个 [ERALIKE游戏](http://www.emuera.net/) \
它的主要表现形式为某种 Ascii Art \
与其他 ERA Game 不同，本游戏中有数千名自由演算的npc \
同时本游戏使用的"AI算法"与通常意义上的AIGC不同，具体技术讨论可以参考这篇[帖子](https://v2ex.com/t/941790) \
目前本游戏对"AI算法"的迭代已经停滞，如果可以提供新的技术方向和想法的话再感谢不过 \
在本游戏中，你可以做任何你想做的事情，游戏设计者也就是我，并没有设计必须完成的强制性的游戏主线 \
由于所有NPC都在自由演算的关系(类似环世界，矮人要塞，模拟人生)，每次新开游戏，都会有所不同 \
可以在[discord频道](discord:https://discord.gg/Hu67GuXkfV)中查看和讨论本游戏具体的设计和更新计划 \
与通常意义的"开源项目"不同，本游戏使用 [cc by nc sa](http://creativecommons.org/licenses/by-nc-sa/2.0/) 协议 \
欢迎每一个人参与到本游戏的开发和设计中，也鼓励每一个人把它改成自己的游戏，本人保留署名权，同时请保持相同方式传播即可 \
请勿将本游戏用于任何"商业用途"

大模型接入
----
游戏现在支持接入大模型来提升游戏体验 \
需在设置面板中手动开启 \
启用 ollama 模式需要玩家自行安装并启动 [ollama](https://ollama.com/) \
ollama 模式下，默认使用 deepseek-r1:1.5b 模型(启用模式时游戏会自动验证和下载)，可在 config.ini 文件中自行修改 \
外部 api 模式需要玩家自行寻找三方 api 服务，例如 openai api \
在 config.ini 文件中填写 base_url api_key 和使用的模型即可 \
游戏过程中使用的提示词也可以在 config.ini 文件中一并修正 \
另外目前大模型技术尚不成熟， ai 模式会造成游戏卡顿， ollama 模式对显卡要求也比较高，作者使用3060进行测试时， deepseek r1 1.5b 每次需要思考20s左右 \
故仅建议尝鲜使用，静候来日

版本说明
----
游戏分为稳定版(Last Version)，和测试版(Test Version)两个版本 \
测试版主要用于测试各种实验性的新系统，确认稳定和兼容性良好后，会被推送到正式版

发展历史
----
eralike游戏引擎 [pyera](https://github.com/qsjl11/pyera) 是故事的开始 \
从一开始计划用 pyera 引擎做简单的小游戏，逐渐迭代到目前的版本 \
从 dieloli 迭代中制作的游戏引擎，发展出的另一个游戏实现是 [erArk](https://github.com/Godofcong-1/erArk)

警告
----
在游戏设计中，有一定的色情，暴力，宗教，等内容，它们的存在是处于某种游戏性和艺术性的考虑 art is freedom.

下载 & 游玩
----
windows & macos系统: 下载[本地址](https://github.com/pokemonchw/dieloli/releases)对应系统最新的压缩文件, 解压后运行game.exe即可 \
linux系统: 下载[源码](https://github.com/pokemonchw/dieloli/archive/refs/heads/master.zip), 并通过pip等工具安装[requirements.txt](https://github.com/pokemonchw/dieloli/blob/master/requirements.txt)中的依赖, 运行game.py即可, 初次运行需要较长时间预热数据

字体
----
本游戏使用了 [Sarasa Mono SC](https://github.com/be5invis/Sarasa-Gothic) 字体，存放于data目录下，出于便于玩家使用的目的，与游戏一同分发

Localization
----
If you want to help with localization, please contact me by email, and I will invite you to join my smartcat

联系方式
----
e-mail:pokemonchw@gmail.com \
discord:https://discord.gg/Hu67GuXkfV

