import pytz
import requests
import pandas as pd
import numpy as np
import xml.etree.ElementTree as et
from datetime import datetime, timedelta


def convert_period_format(date, timezone):
    date_obj = datetime.strptime(date, '%d/%m/%Y')
    date_obj = date_obj.replace(hour=0, minute=0)
    timezone = pytz.timezone(timezone)
    date_obj = timezone.localize(date_obj)
    date_obj = date_obj.astimezone(pytz.utc)
    api_format = date_obj.strftime('%Y%m%d%H%M')
    return api_format


def aggregated_generation(sec_token, start_period, end_period):
    base = 'https://transparency.entsoe.eu/api?'
    security_token = 'securityToken=' + str(sec_token)
    document_type = 'documentType=A75'
    process_type = 'processType=A16'
    in_domain = 'in_Domain=10YCZ-CEPS-----N'
    period_start = 'periodStart=' + str(start_period)
    period_end = 'periodEnd=' + str(end_period)
    api_call = base + security_token + '&' + document_type + '&' + process_type + '&' + in_domain + '&' + period_start +'&' + period_end
    api_answer = requests.get(api_call)
    with open('test.xml', 'w') as f:
        f.write(api_answer.text)
    return


def time_res_to_delta(res):
    res = res.replace('PT', '')
    res = res.replace('M', '')
    res = float(res)
    return timedelta(minutes=res)


def position_to_time(pos, res, start):
    return [start + (res * x) for x in pos]


def type_code_to_text(asset_type):
    codes = pd.read_csv('EUPsrType.csv')
    return codes[codes['Code'] == asset_type]['Meaning'].values[0]

def match_dates(dic):
    index_values = [dic[x][0] for x in dic.keys()]
    common_index_values = list(set.intersection(*map(set, index_values)))
    for x in dic.keys():
        times = np.unique(dic[x][0])
        mask = np.in1d(times,common_index_values)
        mask = np.where(mask)[0]
        dic[x] = dic[x][:,mask]
    return dic

def parse_xml(data):
    tree = et.parse(data)
    root = tree.getroot()
    ns = {'d': 'urn:iec62325.351:tc57wg16:451-6:generationloaddocument:3:0'}
    entsoe_data = {}
    for GenAsset in root.findall('d:TimeSeries', ns):
        asset_type = GenAsset.find('d:MktPSRType', ns)
        asset_type = asset_type.find('d:psrType', ns).text
        asset_type = type_code_to_text(asset_type)
        data = GenAsset.find('d:Period', ns)
        dates = data.find('d:timeInterval', ns)
        start_date = datetime.strptime(dates.find('d:start', ns).text, '%Y-%m-%dT%H:%MZ')
        end_date = datetime.strptime(dates.find('d:end', ns).text, '%Y-%m-%dT%H:%MZ')
        resolution = data.find('d:resolution', ns).text
        resolution = time_res_to_delta(resolution)
        generation = data.find('d:resolution', ns)
        generation = data.findall('d:Point', ns)
        time = [float(x.find('d:position', ns).text) for x in generation]
        time = position_to_time(time, resolution, start_date)
        generation = [float(x.find('d:quantity', ns).text) for x in generation]
        tmp = np.vstack((time,generation))
        if asset_type in entsoe_data:
            tmp2 = entsoe_data.get(asset_type)
            tmp2 = np.hstack((tmp2,tmp))
            entsoe_data[asset_type] = tmp2
        else:
            entsoe_data[asset_type] = tmp
    entsoe_data = match_dates(entsoe_data)
    entsoe_data_pd = pd.DataFrame()
    for asset in entsoe_data.keys():
        entsoe_data_pd[asset] = entsoe_data[asset][1]
        entsoe_data_pd.index = entsoe_data[asset][0]
    entsoe_data_pd.to_csv('entsoeData.csv')
    return


token = '6f7dd5a8-ca23-4f93-80d8-0c6e27533811'
Base = 'https://transparency.entsoe.eu/api'
StartPeriod = '01/01/2016'
EndPeriod = '31/12/2016'
Timezone = 'Europe/Berlin'
Country = 'Germany'

StartPeriod = convert_period_format(StartPeriod, Timezone)
EndPeriod = convert_period_format(EndPeriod, Timezone)
aggregated_generation(token, StartPeriod, EndPeriod)
parse_xml('test.xml')
