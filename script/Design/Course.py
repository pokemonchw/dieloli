from script.Core import TextLoading,ValueHandle,CacheContorl
import math,random

def initPhaseCourseHour():
    '''
    初始化各班级课时
    '''
    phaseCourseTime = TextLoading.getTextData(TextLoading.phaseCourse,'CourseTime')
    primaryWeight = TextLoading.getTextData(TextLoading.phaseCourse,'PrimarySchool')
    juniorMiddleWeight = TextLoading.getTextData(TextLoading.phaseCourse,'JuniorMiddleSchool')
    seniorHighWeight = TextLoading.getTextData(TextLoading.phaseCourse,'SeniorHighSchool')
    nowWeightList = primaryWeight + juniorMiddleWeight + seniorHighWeight
    allClassHourData = {}
    phaseIndex = 0
    for phase in nowWeightList:
        phaseWeightRegin = ValueHandle.getReginList(phase)
        weightMax = 0
        weightMax = sum(map(int,phaseWeightRegin.keys()))
        classHourData = {}
        classHourMax = 0
        if phaseIndex <= 5:
            classHourMax = phaseCourseTime['PrimarySchool']
        elif phaseIndex <= 8:
            classHourMax = phaseCourseTime['JuniorMiddleSchool']
        else:
            classHourMax = phaseCourseTime['SeniorHighSchool']
        for regin in phaseWeightRegin:
            classHourData[phaseWeightRegin[regin]] = math.ceil(classHourMax * (int(regin) / weightMax))
        nowClassHourMax = sum(classHourData.values())
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
        while 1:
            moreHour = 0
            for course in classHourData:
                if moreHour > 0 and classHourData[course] < 14:
                    classHourData[course] += 1
                    moreHour -=1
                elif moreHour >0 and classHourData[course] > 14:
                    moreHour += (classHourData[course] - 14)
                    classHourData[course] -= (classHourData[course] - 14)
            if moreHour == 0:
                break
        allClassHourData[phaseIndex] = classHourData
        phaseIndex += 1
    CacheContorl.courseData['ClassHour'] = allClassHourData
    initPhaseCourseHourExperience()

def initClassTimeTable():
    '''
    初始化各班级课程表
    '''
    courseSession = TextLoading.getGameData(TextLoading.courseSession)
    classTimeTable = {}
    for phase in CacheContorl.courseData['ClassHour']:
        classTime = {}
        classTimeTable[phase] = {}
        classDay = 0
        if phase <= 5:
            classTime = courseSession['PrimarySchool']
            classDay = 6
        elif phase <= 7:
            classTime = courseSession['JuniorMiddleSchool']
            classDay = 7
        else:
            classTime = courseSession['SeniorHighSchool']
            classDay = 8
        classHour = CacheContorl.courseData['ClassHour'][phase]
        classHourIndex = {}
        for course in reversed(list(classHour.keys())):
            classHourIndex.setdefault(course,0)
            while classHourIndex[course] < classHour[course]:
                for day in range(1,classDay):
                    oldDay = day - 1
                    if oldDay == 0:
                        oldDay = classDay - 1
                    classTimeTable[phase].setdefault(day,{})
                    classTimeTable[phase].setdefault(oldDay,{})
                    for i in range(1,len(classTime.keys())):
                        time = list(classTime.keys())[i]
                        if time not in classTimeTable[phase][oldDay] and time not in classTimeTable[phase][day]:
                            classTimeTable[phase][day][time] = course
                            classHourIndex[course] += 1
                            break
                        elif time not in classTimeTable[phase][day]:
                            if course != classTimeTable[phase][oldDay][time]:
                                classTimeTable[phase][day][time] = course
                                classHourIndex[course] += 1
                                break
                            elif i == len(classTime) - 1:
                                classTimeTable[phase][day][time] = course
                                classHourIndex[course] += 1
                                break
                            elif all([k in classTimeTable[phase][day] for k in list(classTime.keys())[i+1:]]):
                                classTimeTable[phase][day][time] = course
                                classHourIndex[course] += 1
                                break
                    if classHourIndex[course] >= classHour[course]:
                        break
    CacheContorl.courseData['ClassTimeTable'] = classTimeTable

def initClassTeacher():
    '''
    初始化各班级任课老师
    '''
    teacherIndex = len(CacheContorl.teacherCourseExperience[list(CacheContorl.teacherCourseExperience.keys())[0]].keys())
    courseMaxA = 0
    courseMaxB = 0
    viceCourseIndexB = 0
    CacheContorl.courseData['ClassTeacher'] = {}
    for phase in CacheContorl.courseData['ClassHour']:
        courseMaxA += len(CacheContorl.courseData['ClassHour'][phase].keys()) * 3
        for course in CacheContorl.courseData['ClassHour'][phase]:
            if CacheContorl.courseData['ClassHour'][phase][course] > 7:
                courseMaxB += 3
            else:
                courseMaxB += 1
                viceCourseIndexB += 1.5
    if teacherIndex >= courseMaxA:
        courseDistributionA()
    elif teacherIndex >= courseMaxB:
        courseDistributionB()

def courseABMainDistribution():
    '''
    课时分配流程AB通用主课时分配流程
    '''
    for phase in range(12,0,-1):
        classList = CacheContorl.placeData['Classroom_' + str(phase)]
        CacheContorl.courseData['ClassTeacher']['Classroom_' + str(phase)] = {}
        for classroom in classList:
            CacheContorl.courseData['ClassTeacher']['Classroom_' + str(phase)][classroom] = {}
            for course in CacheContorl.courseData['ClassHour'][phase - 1]:
                if CacheContorl.courseData['ClassHour'][phase - 1][course] > 7:
                    CacheContorl.courseData['ClassTeacher']['Classroom_' + str(phase)][classroom][course] = []
                    for teacher in CacheContorl.teacherCourseExperience[course]:
                        if teacher not in teacherData:
                            teacherData[teacher] = 0
                            CacheContorl.courseData['ClassTeacher']['Classroom_' + str(phase)][classroom][course].append(teacher)
                            break

teacherData = {}
def courseDistributionA():
    '''
    课时分配流程A
    '''
    courseABMainDistribution()
    for phase in range(1,13):
        classList = CacheContorl.placeData['Classroom_' + str(phase)]
        CacheContorl.courseData['ClassTeacher']['Classroom_' + str(phase)] = {}
        for classroom in classList:
            CacheContorl.courseData['ClassTeacher']['Classroom_' + str(phase)][classroom] = {}
            for course in CacheContorl.courseData['ClassHour'][phase - 1]:
                if CacheContorl.courseData['ClassHour'][phase - 1][course] <= 7:
                    CacheContorl.courseData['ClassTeacher']['Classroom_' + str(phase)][classroom][course] = []
                    for teacher in CacheContorl.teacherCourseExperience[course]:
                        if teacher not in teacherData:
                            teacherData[teacher] = 0
                            CacheContorl.courseData['ClassTeacher']['Classroom_' + str(phase)][classroom][course].append(teacher)
                            break

def courseDistributionB():
    '''
    课时分配流程B
    '''
    courseABMainDistribution()
    for phase in range(1,13):
        classList = CacheContorl.placeData['Classroom_' + str(phase)]
        CacheContorl.courseData['ClassTeacher']['CLassroom_' + str(phase)] = {}
        teacherCourseIndex = 0
        for course in CacheContorl.courseData['ClassHour'][phase - 1]:
            for classroom in classList:
                CacheContorl.courseData['ClassTeacher']['Classroom_' + str(phase)][classroom] = {}
                if CacheContorl.courseData['ClassHour'][phase - 1][course] <= 7:
                    CacheContorl.courseData['ClassTeacher']['Classroom_' + str(phase)][classroom][course] = []
                    for teacher in CacheContorl.teacherCourseExperience[course]:
                        if teacher not in teacherData:
                            CacheContorl.courseData['ClassTeacher']['Classroom_' + str(phase)][classroom][course].append(teacher)
                            teacherCourseIndex += 1
                            if teacherCourseIndex == 2:
                                teacherCourseIndex = 0
                                teacherData[teacher] = 0
                            break

courseKnowledgeData = TextLoading.getGameData(TextLoading.course)
def initPhaseCourseHourExperience():
    '''
    按年级计算各科目课时经验标准量
    '''
    phaseExperience = {}
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

def initCharacterKnowledge():
    '''
    初始化所有角色知识等级
    '''
    for i in CacheContorl.characterData['character']:
        character = CacheContorl.characterData['character'][i]
        characterAge = character.Age
        classGrade = 11
        if characterAge <= 18 and characterAge >= 7:
            classGrade = characterAge - 7
        initExperienceForGrade(classGrade,character)
        CacheContorl.characterData['character'][i] = character
        if characterAge > 18:
            initTeacherKnowledge(character)
            for course in courseKnowledgeData:
                if course not in CacheContorl.teacherCourseExperience:
                    CacheContorl.teacherCourseExperience.setdefault(course,{})
                nowCourseExperience = 0
                for knowledge in courseKnowledgeData[course]['Knowledge']:
                    for skill in courseKnowledgeData[course]['Knowledge'][knowledge]:
                        if knowledge == 'Language':
                            nowCourseExperience += character.Language[skill]
                        else:
                            nowCourseExperience += character.Knowledge[knowledge][skill]
                CacheContorl.teacherCourseExperience[course][i] = nowCourseExperience

def initTeacherKnowledge(character) -> dict:
    '''
    按年龄修正教师知识等级
    '''
    characterAge = character.Age
    studyYear = characterAge - 18
    for knowledge in character.Knowledge:
        for skill in character.Knowledge[knowledge]:
            character.Knowledge[knowledge][skill] += character.Knowledge[knowledge][skill] / 12 * studyYear * random.uniform(0.25,0.75)
    for language in character.Language:
        character.Knowledge[knowledge][skill] += character.Language[language] / 12 * studyYear * random.uniform(0.25,0.75)

courseData = TextLoading.getGameData(TextLoading.course)
def initExperienceForGrade(classGrade:str,character):
    '''
    按年级生成角色初始经验数据
    '''
    phaseExperienceData = CacheContorl.courseData['PhaseExperience']
    for garde in range(classGrade):
        experienceData = phaseExperienceData[garde]
        for knowledge in experienceData:
            if knowledge == 'Language':
                for skill in experienceData[knowledge]:
                    skillExperience = experienceData[knowledge][skill]
                    skillInterest = character.Interest[skill]
                    skillExperience *= skillInterest
                    if skill in character.Language:
                        character.Language[skill] += skillExperience
                    else:
                        character.Language[skill] = skillExperience
            else:
                if knowledge not in character.Knowledge:
                    character.Knowledge.setdefault(knowledge,{})
                for skill in experienceData[knowledge]:
                    skillExperience = experienceData[knowledge][skill]
                    skillInterest = character.Interest[skill]
                    skillExperience *= skillInterest
                    if skill in character.Knowledge[knowledge]:
                        character.Knowledge[knowledge][skill] += skillExperience
                    else:
                        character.Knowledge[knowledge][skill] = skillExperience

