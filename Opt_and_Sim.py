import urllib.request
import ssl
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import ast

# Ignore SSL certificate errors
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

# TRAIN API
response = urllib.request.urlopen('https://hackatraintrainapi.azurewebsites.net/api/train/ns/Ehv/1117?fakeData=1',
                                   context=ctx)

data_t = response.read().decode()
df_t = pd.read_json(data_t)

sitting = np.array([row['totalPassengers'] for row in df_t['coaches']])

# PLATFORM API
response = urllib.request.urlopen('https://trainapiplatform.azurewebsites.net/api/TrainPlatformCapacity?totalCoaches=6&maxPassengers=110',
                                   context=ctx)

x = response.read().decode()
plat_arrival = np.array(ast.literal_eval(x))

print(len(plat_arrival))

# OPTIMIZATION FUNCTION
def optimize(sit, plat):
    ''' This function takes the 1) compartment and 2) platform occupation
    as an input, and returns 1) the optimal area to put the train,
    2) the new compartment occupation resulting from the chosen train arrival
    4) the associated cost (how far from evenly distributed), and 5) how people
    are pressed together when entering the train'''

    length_train = len(sit)
    length_plat = len(plat)

    num_options = length_plat-length_train+1
    cost = np.zeros((num_options,))*np.nan

    new_sits = []
    plat_news = []

    # for every option that the train can stop,
    # compute the cost (how far from ideal)
    for i in range(0,num_options):
        plat_new = plat.copy()

        # compress platform
        plat_compr_left = plat_new[:i].sum()
        plat_compr_right = plat_new[length_train+i:].sum()
        plat_new[i] = plat_new[i]+plat_compr_left
        plat_new[length_train+i-1] = plat_new[length_train+i-1]+plat_compr_right
        plat_new = plat_new[i:i+length_train]
        plat_news.append(plat_new)

        # compute new distribution
        new_sit = sit+plat_new
        new_sits.append(new_sit)

        # compute cost function
        ideal = new_sit.mean()
        cost[i] = np.abs(new_sit-ideal).sum()

    # find the best option
    best_opt = np.argmin(cost)
    best_cost = np.min(cost)

    return (best_opt, best_opt+length_train), new_sits[best_opt], best_cost, plat_news[best_opt]

print(optimize(sitting, plat_arrival))
