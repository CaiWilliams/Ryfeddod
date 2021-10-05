import difflib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil import tz
import os
import json
import pickle
import pytz
import requests
import io
from pvlive_api import PVLive
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import copy
import xml.etree.ElementTree as et

class Grid:

    def __init__(self, MixDir):
        self.BMRSKey = "zz6sqbg3mg0ybyc"
        self.ENTSOEKey = "6f7dd5a8-ca23-4f93-80d8-0c6e27533811"

        with open(MixDir) as Mix_File:
            self.Mix = json.load(Mix_File)

        datasources = set()
        for Tech in self.Mix["Technologies"]:
            datasources.add(Tech["Source"])

        self.StartDate = datetime.strptime(self.Mix['StartDate'], '%Y-%m-%d')
        self.EndDate = datetime.strptime(self.Mix['EndDate'], '%Y-%m-%d')
        self.timezone = 'Europe/Prague'

        if self.Mix["Country"] == "UUK":
            if "BMRS" in datasources:
                self.BMRSFetch()
            if "PVLive" in datasources:
                self.PVLiveFetch()

