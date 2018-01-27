import script.TextLoading as textload

flowContorl = {'restartGame': 0, 'quitGame': 0}
wframeMouse = {'wFrameUp': 2, 'mouseRight': 0, 'mouseLeaveCmd': 1, 'wFrameLinesUp': 2, 'wFrameLineState': 2,
               'wFrameRePrint': 0}
cmd_map = {}
playObject = {'objectId': '', 'object': {}}
temObjectDefault = textload.loadRoleAtrText('Default')
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
                      'SexGrade':temObjectDefault['SexGrade']
                      }
featuresList = {'Age':"","Chastity":"",'Disposition':"",'SelfConfidence':"",'Friends':"",
                'Figure':"",'Sex':"",'AnimalInternal':"",'AnimalExternal':"",'Charm':""}

temporaryObject = {}

inputCache = ['']

inputPosition = {'position': 0}

outputTextStyle = 'standard'

textStylePosition = {'position':0}

textStyleCache = ['standard']

textOneByOneRichCache = {'textList':[],'styleList':[]}