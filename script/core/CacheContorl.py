from core import TextLoading

# 流程用变量组
flowContorl = {}

# 主页监听控制流程用变量组
wframeMouse = {}

# cmd存储
cmd_map = {}

# 角色对象数据缓存组
playObject = {}

# 默认属性模板数据读取
temObjectDefault = TextLoading.getTextData(TextLoading.roleId,'Default')

# 默认属性模板数据备份
temporaryObjectBak = {'Name':temObjectDefault['Name'],
                      'NickName':temObjectDefault['NickName'],
                      'SelfName':temObjectDefault['SelfName'],
                      'Species':temObjectDefault['Species'],
                      'Relationship':temObjectDefault['Relationship'],
                      'Sex':temObjectDefault['Sex'],
                      'Age':temObjectDefault['Age'],
                      'San':temObjectDefault['San'],
                      'Intimate':temObjectDefault['Intimate'],
                      'Graces':temObjectDefault['Graces'],
                      'Features':temObjectDefault['Features'],
                      'HitPointMax':temObjectDefault['HitPointMax'],
                      'HitPoint':temObjectDefault['HitPointMax'],
                      'ManaPointMax':temObjectDefault['ManaPointMax'],
                      'ManaPoint':temObjectDefault['ManaPointMax'],
                      'SexExperience':temObjectDefault['SexExperience'],
                      'SexGrade':temObjectDefault['SexGrade'],
                      'Engraving':temObjectDefault['Engraving'],
                      'Clothing':temObjectDefault['Clothing'],
                      'SexItem':temObjectDefault['SexItem'],
                      'Gold':temObjectDefault['Gold'],
                      'Position':temObjectDefault['Position']
                      }

# 素质数据临时缓存
featuresList = {}

# 临时角色数据控制对象
temporaryObject = {}

# 输入记录（最大20）
inputCache = []

# 回溯输入记录用定位
inputPosition = {}

# 富文本记录输出样式临时缓存
outputTextStyle = ''

# 富文本回溯样式记录用定位
textStylePosition = {}

# 富文本样式记录
textStyleCache = []

# 富文本精确样式记录
textOneByOneRichCache = {}

lastcursor = [0]

# 图片id
imageid = 0

# cmd数据
cmdData = {}

# 游戏时间
gameTime = {}

# 面板状态
panelState = {}

# 存档页面最大数量
maxSavePage = 0

textWait = 0

sceneData = {}

mapData = {}