from script.Core import CacheContorl,TextLoading
import random

def initCharacterInterest():
    '''
    初始化全部角色兴趣/精力/天赋数值分配
    '''
    interestList = []
    languageSkills = TextLoading.getGameData(TextLoading.languageSkillsPath)
    interestList += list(languageSkills.keys())
    knowledgeData = TextLoading.getGameData(TextLoading.knowledge)
    for knowledgeTag in knowledgeData:
        knowledgeList = knowledgeData[knowledgeTag]
        interestList += list(knowledgeList['Knowledge'].keys())
    interestAverage = 100 / len(interestList)
    for character in CacheContorl.characterData['character']:
        nowInterestValueMax = 100
        nowInterestList = interestList.copy()
        #numpy.random.shuffle(nowInterestList)
        for interest in nowInterestList:
            if interest != nowInterestList[-1]:
                nowInterestAverage = nowInterestValueMax / len(nowInterestList)
                nowInterValue = nowInterestAverage * random.uniform(0.75,1.25)
                nowInterestValueMax -= nowInterValue
                CacheContorl.characterData['character'][character]['Interest'][interest] = nowInterValue / interestAverage
            else:
                CacheContorl.characterData['character'][character]['Interest'][interest] = nowInterestValueMax
