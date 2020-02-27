from script.Core import CacheContorl,TextLoading,ValueHandle
from script.Design import AttrCalculation,Clothing,Nature
import random
import uuid
import time

class Character(object):

    def __init__(self):
        self.Name = '主人公'
        self.NickName = '你'
        self.SelfName = '我'
        self.Species = '人类'
        self.Sex = 'Man'
        self.Age = 17
        self.Relationship = '无'
        self.Intimate = 0
        self.Graces = 0
        self.Features = {}
        self.HitPointMax = 0
        self.HitPoint = 0
        self.ManaPointMax = 0
        self.ManaPoint = 0
        self.SexExperience = {}
        self.SexGrade = {}
        self.State = 'arder'
        self.Engraving = {}
        self.Clothing = {
            "Coat":{},
            "Underwear":{},
            "Pants":{},
            "Skirt":{},
            "Shoes":{},
            "Socks":{},
            "Bra":{},
            "Underpants":{}
        }
        self.SexItem = {}
        self.Item = []
        self.Height = {}
        self.Weight = {}
        self.BodyFat = {}
        self.Measurements = {}
        self.Behavior = {}
        self.Gold = 0
        self.Position = ['0']
        self.Class = []
        self.Office = ['0']
        self.Knowledge = {}
        self.Language = {}
        self.MotherTongue = "Chinese"
        self.Interest = {}
        self.Dormitory = '0'
        self.Birthday = {}
        self.WeigtTem = 'Ordinary'
        self.BodyFatTem = 'Ordinary'
        self.BodyFat = {}
        self.SexExperienceTem = 'None'
        self.ClothingTem = 'Uniform'
        self.ChestTem = 'Ordinary'
        self.Chest = {}
        self.Nature = {}
        self.Status = {}
        self.PutOn = {}
        self.WearItem = {}
        self.HitPointTem = 'Ordinary'
        self.ManaPointTem = 'Ordinary'
        self.SocialContact = {}

    def initAttr(self):
        '''
        随机生成角色属性
        '''
        self.Language[self.MotherTongue] = 10000
        self.Birthday = AttrCalculation.getRandNpcBirthDay(self.Age)
        self.Height = AttrCalculation.getHeight(self.Sex,self.Age,{})
        bmi = AttrCalculation.getBMI(self.WeigtTem)
        self.Weight = AttrCalculation.getWeight(bmi,self.Height['NowHeight'])
        self.BodyFat = AttrCalculation.getBodyFat(self.Sex,self.BodyFatTem)
        self.Measurements = AttrCalculation.getMeasurements(self.Sex,self.Height['NowHeight'],self.Weight,self.BodyFat,self.BodyFatTem)
        self.SexExperience = AttrCalculation.getSexExperience(self.SexExperienceTem)
        self.SexGrade = AttrCalculation.getSexGrade(self.SexExperience)
        defaultClothingData = Clothing.creatorSuit(self.ClothingTem,self.Sex)
        self.Clothing = {clothing:{uuid.uuid1():defaultClothingData[clothing]} if clothing in defaultClothingData else {} for clothing in self.Clothing}
        self.Chest = AttrCalculation.getChest(self.ChestTem,self.Birthday)
        self.HitPointMax = AttrCalculation.getMaxHitPoint(self.HitPointTem)
        self.HitPoint = self.HitPointMax
        self.ManaPointMax = AttrCalculation.getMaxManaPoint(self.ManaPointTem)
        self.ManaPoint = self.ManaPointMax
        self.Nature = Nature.getRandomNature()
        self.Status = TextLoading.getGameData(TextLoading.characterStatePath)
        self.WearItem = {
            "Wear":{key:'' for key in TextLoading.getGameData(TextLoading.wearItemPath)['Wear']},
            "Item":{}
        }
        self.Engraving = {
            "Pain":0,
            "Happy":0,
            "Yield":0,
            "Fear":0,
            "Resistance":0
        }
        self.SocialContact = {social:{} for social in TextLoading.getTextData(TextLoading.stageWordPath,'144')}
        self.initClass()

    def initClass(self):
        '''
        初始化角色班级
        '''
        if self.Age <= 18 and self.Age >= 7:
            classGrade = str(self.Age - 6)
            self.Class = random.choice(CacheContorl.placeData['Classroom_' + classGrade])
