# -*- coding: utf-8 -*-
#Data_analytics_a2.ipynb
# author : Abhishek Kumar

import numpy as np
import pandas as pd
import random
from scipy.optimize import minimize
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
import random

def read_dataset(filename):
    df = pd.read_csv(filename)
    Zodiac_index = df['ZodiacIndex'].values
    Degree = df['Degree'].values
    Minute = df['Minute'].values
    Second = df['Second'].values
    sun = Zodiac_index, Degree, Minute, Second
    Avg_Zodiac_index = df['ZodiacIndexAverageSun'].values
    DegreeMean = df['DegreeMean'].values
    MinuteMean = df['MinuteMean'].values
    SecondMean = df['SecondMean'].values
    avg_sun = Avg_Zodiac_index, DegreeMean, MinuteMean, SecondMean
    return sun, avg_sun

def calc_longitude(z,d,m,s):
    return (z*30 + d + m/60 + s/3600 ) * math.pi/180

def longitude(filename):
    sun,avg_sun = read_dataset(filename)
    Zodiac_index, Degree, Minute, Second = sun
    Avg_Zodiac_index, DegreeMean, MinuteMean, SecondMean = avg_sun
    alpha = []
    beta = []
    for i in range(len(Zodiac_index)):
        a = calc_longitude(Zodiac_index[i],Degree[i],Minute[i],Second[i])
        b = calc_longitude(Avg_Zodiac_index[i],DegreeMean[i],MinuteMean[i],SecondMean[i])
        alpha.append(a)
        beta.append(b)
    return alpha, beta

def calc_coordinates(a,b,alpha,beta):
    x_coord = (math.cos(b)*(alpha + beta * a) - math.sin(b) * (1 + a)) / (alpha - beta)
    y_coord = (alpha * beta * math.cos(b)*(a + 1) - (math.sin(b)*(alpha*a + beta))) / (alpha - beta)
    return x_coord, y_coord

def coordinates(a,b,alpha,beta):
    x_coord = []
    y_coord = []
    for i in range(len(alpha)):
        x, y = calc_coordinates(a,b,math.tan(alpha[i]),math.tan(beta[i]))
        x_coord.append(x)
        y_coord.append(y)
    return x_coord, y_coord

def arithmetic_mean(x,y):
    mean = 0
    for i in range(len(x)):
        mean += math.sqrt(x[i]**2 + y[i]**2)
    return mean/len(x)

def geometric_mean(x,y):
    mean = 0
    for i in range(len(x)):
        distance = math.sqrt(x[i]**2 + y[i]**2)
        mean += math.log(distance)
    mean /= len(x)
    return math.exp(mean)

def calc_loss(params,alpha,beta):
    a, b = params
    x_coord, y_coord = coordinates(a, b, alpha, beta)
    a_mean = arithmetic_mean(x_coord,y_coord)
    g_mean = geometric_mean(x_coord,y_coord)
    return abs(math.log(a_mean) - math.log(g_mean))

def minimize_loss(alpha,beta,a,b):
    params = a, b
    parameter = minimize(calc_loss,params,args = (alpha,beta),method = 'L-BFGS-B')
    #parameter = minimize(calc_loss,params,args = (alpha,beta),method = 'TNC')
    #print(parameter)
    opt_param = parameter['x']
    loss = parameter['fun']
    return opt_param,loss

if __name__ == '__main__':
    alpha, beta = longitude('../data/01_data_mars_opposition.csv')
    opt_param, loss = minimize_loss(alpha,beta,5,3)
    print("Value of x is : ",opt_param[0])
    print("Value of y is : ",opt_param[1])
    print("Total Loss is : ",loss)
    x_coord, y_coord = coordinates(opt_param[0], opt_param[1], alpha, beta)
    print("The coordinates of Mars position according to calculated x and y are as follows:")
    for i in range(len(x_coord)):
    	print("(",x_coord[i],",",y_coord[i],")")
    	print
    rad = math.sqrt(x_coord[1]**2 + y_coord[1]**2)
    print(rad)
    ax = plt.gca()
    ax.cla()
    circle = plt.Circle((0,0), radius= rad, color='g', fill=False)
    ax.set_xlim((-12, 12))
    ax.set_ylim((-12, 12))
    plt.scatter(x_coord,y_coord)
    plt.gcf().gca().add_artist(circle)
    plt.show()

# loss for a = 5 and b =3 is [0.96787627 2.5983634 ] 0.004097728218914742 
# loss for a = 10 and b = 11 is [10.13478687 -9.97575273] 0.0050076389416142675
# loss for a = 3 and b = 3 is [0.96788021 2.59836291] 0.004097728219773611
# loss for a = 1 and b = 1 is [ 0.91283251 -0.54297648] 0.00409890443662686
# loss for a = 100 and b = 80 is [100.00001467  81.12885054] 0.005374375952525767

