from script.Design import CharacterBehavior,GameTime

def gameUpdateFlow():
    '''
    游戏流程刷新
    '''
    GameTime.initSchoolCourseTimeStatus()
    CharacterBehavior.initCharacterBehavior()
