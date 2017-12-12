import script.TextLoading as textload

flowContorl = {'restartGame': 0, 'quitGame': 0}
wframeMouse = {'wFrameUp': 2, 'mouseRight': 0, 'mouseLeaveCmd': 1, 'wFrameLinesUp': 2, 'wFrameLineState': 2,
               'wFrameRePrint': 0}
cmd_map = {}
playObject = {'objectId': '', 'object': {}}
temObjectDefault = textload.loadRoleAtrText('Default')
temporaryObjectBak = {'Name': temObjectDefault['Name'],
                           'NickName': temObjectDefault['NickName'],
                           'Sex': temObjectDefault['Sex'],
                           'Age': temObjectDefault['Age']}
temporaryObject = {}

inputCache = ['']

inputPosition = {'position': 0}
