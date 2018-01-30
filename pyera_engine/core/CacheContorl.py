import script.TextLoading as textload

# 流程用变量组
flowContorl = {'restartGame': 0, 'quitGame': 0}

# 主页监听控制流程用变量组
wframeMouse = {'wFrameUp': 2, 'mouseRight': 0, 'mouseLeaveCmd': 1, 'wFrameLinesUp': 2, 'wFrameLineState': 2,
               'wFrameRePrint': 0}

# cmd存储
cmd_map = {}

# 角色对象数据缓存组
playObject = {'objectId': '', 'object': {}}

# 默认属性模板数据读取
temObjectDefault = textload.loadRoleAtrText('Default')

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
                      'Engraving':temObjectDefault['Engraving']
                      }

# 素质数据临时缓存
featuresList = {'Age':"","Chastity":"",'Disposition':"",'Courage':"",'SelfConfidence':"",'Friends':"",
                'Figure':"",'Sex':"",'AnimalInternal':"",'AnimalExternal':"",'Charm':""}

# 临时角色数据控制对象
temporaryObject = {}

# 输入记录（最大20）
inputCache = ['']

# 回溯输入记录用定位
inputPosition = {'position': 0}

# 富文本记录输出样式临时缓存
outputTextStyle = 'standard'

# 富文本回溯样式记录用定位
textStylePosition = {'position':0}

# 富文本样式记录
textStyleCache = ['standard']

# 富文本精确样式记录
textOneByOneRichCache = {'textList':[],'styleList':[]}