from script.Core import TextLoading,ValueHandle
import math

# 初始化各班级课时和任课老师
def initCourse():
    phaseCourseTime = TextLoading.getTextData(TextLoading.phaseCourse,'CourseTime')
    primaryWeight = TextLoading.getTextData(TextLoading.phaseCourse,'PrimarySchool')
    juniorMiddleWeight = TextLoading.getTextData(TextLoading.phaseCourse,'JuniorMiddleSchool')
    seniorHighWeight = TextLoading.getTextData(TextLoading.phaseCourse,'SeniorHighSchool')
    nowWeightList = primaryWeight + juniorMiddleWeight + seniorHighWeight
    nowWeightList.reverse()
    phaseIndex = 0
    allClassHourData = {}
    for phase in nowWeightList:
        weightMax = 0
        phaseWeightRegin = ValueHandle.getReginList(phase)
        for regin in phaseWeightRegin:
            weightMax += int(regin)
        classHourData = {}
        classHourMax = 0
        if phaseIndex <= 2:
            classHourMax = phaseCourseTime['SeniorHighSchool']
        elif phaseIndex <= 5:
            classHourMax = phaseCourseTime['JuniorMiddleSchool']
        else:
            classHourMax = phaseCourseTime['PrimarySchool']
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
