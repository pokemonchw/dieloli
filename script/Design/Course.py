from script.Core import TextLoading,ValueHandle,CacheContorl
import math,datetime,random

# 初始化各班级课时和任课老师
def initPhaseCourseHour():
    phaseCourseTime = TextLoading.getTextData(TextLoading.phaseCourse,'CourseTime')
    primaryWeight = TextLoading.getTextData(TextLoading.phaseCourse,'PrimarySchool')
    juniorMiddleWeight = TextLoading.getTextData(TextLoading.phaseCourse,'JuniorMiddleSchool')
    seniorHighWeight = TextLoading.getTextData(TextLoading.phaseCourse,'SeniorHighSchool')
    nowWeightList = primaryWeight + juniorMiddleWeight + seniorHighWeight
    allClassHourData = {}
    phaseIndex = 0
    for phase in nowWeightList:
        weightMax = 0
        phaseWeightRegin = ValueHandle.getReginList(phase)
        for regin in phaseWeightRegin:
            weightMax += int(regin)
        classHourData = {}
        classHourMax = 0
        if phaseIndex <= 5:
            classHourMax = phaseCourseTime['PrimarySchool']
        elif phaseIndex <= 7:
            classHourMax = phaseCourseTime['JuniorMiddleSchool']
        else:
            classHourMax = phaseCourseTime['SeniorHighSchool']
        for regin in phaseWeightRegin:
            classHourData[phaseWeightRegin[regin]] = math.ceil(classHourMax * (int(regin) / weightMax))
        nowClassHourMax = 0
        for course in classHourData:
            nowClassHourMax += classHourData[course]
        while nowClassHourMax != classHourMax:
            for course in classHourData:
                if nowClassHourMax == classHourMax:
                    break
                elif classHourData[course] > 1 and nowClassHourMax > classHourMax:
                    classHourData[course] -= 1
                    nowClassHourMax -= 1
                elif nowClassHourMax < classHourMax:
                    classHourData[course] += 1
                    nowClassHourMax += 1
        allClassHourData[phaseIndex] = classHourData
        phaseIndex += 1
    CacheContorl.courseData['ClassHour'] = allClassHourData
    initPhaseCourseHourExperience()

# 初始化每年级科目课时经验标准量
def initPhaseCourseHourExperience():
    phaseExperience = {}
    courseKnowledgeData = TextLoading.getGameData(TextLoading.course)
    for phase in CacheContorl.courseData['ClassHour']:
        phaseExperience[phase] = {}
        for course in CacheContorl.courseData['ClassHour'][phase]:
            courseHour = CacheContorl.courseData['ClassHour'][phase][course]
            for knowledge in courseKnowledgeData[course]['Knowledge']:
                if knowledge not in phaseExperience[phase]:
                    phaseExperience[phase][knowledge] = {}
                for skill in courseKnowledgeData[course]['Knowledge'][knowledge]:
                    skillExperience = courseKnowledgeData[course]['Knowledge'][knowledge][skill] * courseHour * 38
                    if skill in phaseExperience[phase][knowledge]:
                        phaseExperience[phase][knowledge][skill] += skillExperience
                    else:
                        phaseExperience[phase][knowledge][skill] = skillExperience
    CacheContorl.courseData['PhaseExperience'] = phaseExperience

# 初始化角色知识等级
def initCharacterKnowledge():
    for i in CacheContorl.characterData['character']:
        character = CacheContorl.characterData['character'][i]
        characterInterestData = character['Interest']
        characterAge = character['Age']
        classGrade = 11
        if characterAge <= 18 and characterAge >= 7:
            classGrade = characterAge - 7
        initExperienceForGrade(classGrade,character)
        CacheContorl.characterData['character'][i] = character
        if characterAge > 18:
            initTeacherKnowledge(character)

def initTeacherKnowledge(character):
    characterAge = character['Age']
    studyYear = characterAge - 18
    for knowledge in character['Knowledge']:
        for skill in character['Knowledge'][knowledge]:
            character['Knowledge'][knowledge][skill] += character['Knowledge'][knowledge][skill] / 12 * studyYear * random.uniform(0.25,0.75)
    for language in character['Language']:
        character['Knowledge'][knowledge][skill] += character['Language'][language] / 12 * studyYear * random.uniform(0.25,0.75)

courseData = TextLoading.getGameData(TextLoading.course)
def initExperienceForGrade(classGrade,character):
    phaseExperienceData = CacheContorl.courseData['PhaseExperience']
    for garde in range(classGrade):
        experienceData = phaseExperienceData[garde]
        for knowledge in experienceData:
            if knowledge == 'Language':
                for skill in experienceData[knowledge]:
                    skillExperience = experienceData[knowledge][skill]
                    skillInterest = character['Interest'][skill]
                    skillExperience *= skillInterest
                    if skill in character['Language']:
                        character['Language'][skill] += skillExperience
                    else:
                        character['Language'][skill] = skillExperience
            else:
                character['Knowledge'].setdefault(knowledge,{})
                for skill in experienceData[knowledge]:
                    skillExperience = experienceData[knowledge][skill]
                    skillInterest = character['Interest'][skill]
                    skillExperience *= skillInterest
                    if skill in character['Knowledge'][knowledge]:
                        character['Knowledge'][knowledge][skill] += skillExperience
                    else:
                        character['Knowledge'][knowledge][skill] = skillExperience
