# -*- coding: utf-8 -*-
'''
Created on Sat Jan 11 19:38:27 2020

@author: sparkbyexamples.com
'''


import numpy as np
import pandas as pd
from pandas_profiling import ProfileReport

df = pd.DataFrame(np.random.rand(100, 5), columns=["a", "b", "c", "d", "e"])

profile = ProfileReport(df, title="Pandas Profiling Report")
profile.to_file("your_report.html")

# # As a JSON string
# json_data = profile.to_json()

# As a file
profile.to_file("your_report.json")

aa = 1