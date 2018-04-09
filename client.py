# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 06:03:31 2018

@author: Larry Lai
"""
import numpy as np

from clarifai.rest import ClarifaiApp
from clarifai.rest import Image as ClImage

def get_tags(model, image_url, threshold = 0.88):
    response = model.predict_by_url(image_url)
    tags = []
    for concept in response['outputs'][0]['data']['concepts']:
        if concept['value'] >= threshold:
            tags.append(concept['name'])
    return tags

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import load_model

import pickle

def load_obj(name):
	with open(name + '.pkl', 'rb+') as f:
		return pickle.load(f)

regressor = load_model('my_model.h5')

nfollowers = input('Enter the number of followers you have:')
nfollowing = input('Enter the number of people you are following:')
nposts = input('Enter the total number of posts you have:')
hashtags = '0'
url = input('Enter a url to the image:')
nfollowers = int(int(nfollowers)/50)


app = ClarifaiApp(api_key='e03f648f9f84485e96008090c27eacd3')
model = app.models.get('general-v1.3')
tags = get_tags(model, url)
all_tags = load_obj('4orabovetags')

#       followers, following, nposts, ntags, time, ...n features..., likes
x = [nfollowers, nfollowing, nposts, 0];
for tag in all_tags:
    if tag in tags:
        x.append(1)
    else:
        x.append(0)

X = np.array(x, 'float64')
X = np.reshape(X, (1, X.shape[0]))
from sklearn.preprocessing import StandardScaler
sc = load_obj('sc')
X = sc.transform(X)

y = regressor.predict(X)



print ('The predicted number of likes is \n')
print (int(y[0][0]))




            


