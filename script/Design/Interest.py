from script.Core import CacheContorl,TextLoading
import numpy,datetime,random

# 初始化角色兴趣值
def initCharacterInterest():
    interestList = []
    languageSkills = TextLoading.getGameData(TextLoading.languageSkillsPath)
    interestList += list(languageSkills.keys())
    knowledgeData = TextLoading.getGameData(TextLoading.knowledge)
    for knowledgeTag in knowledgeData:
        knowledgeList = knowledgeData[knowledgeTag]
        interestList += list(knowledgeList['Knowledge'].keys())
    interestAverage = 100 / len(interestList)
    for character in CacheContorl.characterData['character']:
        time1 = datetime.datetime.now()
        nowInterestValueMax = 100
        nowInterestList = interestList.copy()
        numpy.random.shuffle(nowInterestList)
        for interest in nowInterestList:
            if interest != nowInterestList[-1]:
                nowInterestAverage = nowInterestValueMax / len(nowInterestList)
                nowInterValue = nowInterestAverage * random.uniform(0.75,1.25)
                nowInterestValueMax -= nowInterValue
                CacheContorl.characterData['character'][character]['Interest'][interest] = nowInterValue / interestAverage
            else:
                CacheContorl.characterData['character'][character]['Interest'][interest] = nowInterestValueMax
        time4 = datetime.datetime.now()
