import pandas as pd
import numpy as np
import re


def ENTSOECodes(domain_name):
    E = pd.read_csv('C:\\Users\Cai Williams\PycharmProjects\Ryfeddod\Data\ENTSOELocations.csv')
    N = ['Name 0', 'Name 1', 'Name 2']

    for X in N:
        EX = E[X].dropna().to_list()
        DN = [string for string in EX if domain_name in string]
        if len(DN) > 0:
            break

    Code = E[E[X].isin([DN[0]])]['Code'].values[0]
    return Code

print(ENTSOECodes('Turkey'))