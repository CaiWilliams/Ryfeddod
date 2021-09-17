import pytz
import requests
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

def Convert_Period_Format(Date,Timezone):
    Date_Obj = datetime.strptime(Date, '%d/%M/%Y')
    Date_Obj = Date_Obj.replace(hour=0, minute=0)
    timezone = pytz.timezone(Timezone)
    Date_Obj = timezone.localize(Date_Obj)
    Date_Obj = Date_Obj.astimezone(pytz.utc)
    API_Format = Date_Obj.strftime('%Y%m%d%H%M')
    return API_Format

def Aggragated_Generation(token,StartPeriod,EndPeriod):
    Base = 'https://transparency.entsoe.eu/api?'
    SecurityToken='securityToken='+ str(token)
    DocumentType = 'documentType=A75'
    ProcessType = 'processType=A16'
    in_Domain = 'in_Domain=10YCZ-CEPS-----N'
    PeriodStart = 'periodStart=' + str(StartPeriod)
    PeriodEnd = 'periodEnd=' + str(EndPeriod)
    APICall = Base + SecurityToken + '&' + DocumentType + '&' + ProcessType + '&' + in_Domain + '&' + PeriodStart + '&' + PeriodEnd
    APIAnswer = requests.get(APICall)
    #APIText = APIAnswer.text()
    with open('test.xml','w') as f:
        f.write(APIAnswer.text)
    return

def Parse_XML(Data):
    tree = ET.parse(Data)
    root = tree.getroot()
    ns = {'d:','urn:iec62325.351:tc57wg16:451-6:generationloaddocument:3:0'}
    print([elem.tag for elem in root.iter('objectAggregation')])
    print(root.findall('d://TimeSeries/objectAggregation',ns))
    for GenAsset in root.findall('TimeSeries'):
        print('AA')
        print(GenAsset.find('inBiddingZone_Domain.mRID'))
    return

token = '6f7dd5a8-ca23-4f93-80d8-0c6e27533811'
Base = 'https://transparency.entsoe.eu/api'
StartPeriod = '01/01/2016'
EndPeriod = '31/12/2016'
Timezone = 'Europe/Berlin'
Country = 'Germany'

#StartPeriod = Convert_Period_Format(StartPeriod,Timezone)
#EndPeriod = Convert_Period_Format(EndPeriod,Timezone)
#Aggragated_Generation(token,StartPeriod,EndPeriod)
Parse_XML('test.xml')


