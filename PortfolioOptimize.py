# Easy Tool for Portfolio Optimization

import math
import datetime as dt
import pandas as pd
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as pdr
from scipy.optimize import fmin

# 1. User Input

tick = input("Please input a ticker:")
ticker = []
while tick != "":
    ticker.append(tick)
    tick = input("Please input a ticker, press ENTER to stop:")

# ticker = ["IBM","WMT","C"]

# Quit if input tickers are not enough
if len(ticker) <= 1:
    print("Need more tickers input!")
    quit()

startyear = input("Please input the start year:")
if startyear.isdigit():
    if eval(startyear) >= 1985 and eval(startyear) < 2018:
        pass
    else:
        print("Time period not available!")
        quit()
else:
    print("Time period not available!")
    quit()

start = dt.datetime(eval(startyear),1,1) # Start date
end = dt.datetime(2018,1,1) # End date
return_type = "BMS" # "BMS" for business monthly start date
riskfree = 0.0003 # Set a risk free interest rate

# 2. Functions definition

# Function 1: accessing prices data
def getprice(ticker,start,end):
    data = pdr.DataReader(ticker,"morningstar",start,end)
    price = data["Close"]
    p = []
    for i in range(1,len(price)):
        p.append(price[i])
    df_price = pd.DataFrame(p,columns=[ticker])
    return df_price

# Function 2: accessing returns data
def getret(ticker,df_price):
    price = df_price[ticker]
    simret = price.pct_change(1)
    # logret = []
    # for k in range(len(price)-1):
    #    logret.append(log(price[k+1]/price[k]))
    ret = []
    for i in range(1,len(simret)):
        ret.append(simret[i])
    df_ret = pd.DataFrame(ret,columns=[ticker])
    return df_ret

# Function 3: portfolio variance calculation (actually, the return is standard deviation)
def getvar(covmat,weight): # length of weight = length of ticker - 1
    allweight = sp.append(weight,1-sum(weight))
    var_total = 0
    for i in range(len(covmat)):
        for j in range(len(covmat)):
            if all(w>=0 for w in allweight):
                var_total += allweight[i]*allweight[j]*covmat[i][j]
            else:
                var_total += 9999
    std_total = math.sqrt(var_total)
    return std_total

# Function 4: sharp ratio calculation
def getsharp(ret,rf,weight,covmat):
    allweight = sp.append(weight,1-sum(weight))
    matret = np.array(ret)*allweight
    portret = sum((matret+1).prod(axis=0)-1)*12/len(ret)
    # meanret = sp.mean(ret,axis=0)
    # portret = sp.dot(sp.array(meanret),allweight)
    std = getvar(covmat,weight)
    sharp = (portret-rf)/std
    return sharp

# Function 5: min variance function for fmin() optimization
# Be careful! outside variable: covmat
def testvar(weight): # length of weight = length of ticker - 1
    portvar = getvar(covmat,weight)
    return portvar

# Function 6: max sharp function for fmin() optimization
# Be careful! outside variables: ret_all, riskfree, covmat
def testsharp(weight):
    portsharp = -getsharp(ret_all,riskfree,weight,covmat)
    # Since it use fmin() to find the max sharp, the sharp should be a negative value
    return portsharp

# 3. Data Accessing

# Get monthly data from MorningStar
price_w = pd.DataFrame()
price_all = pd.DataFrame()
for t in ticker:
    p_ticker = pdr.DataReader(t,"morningstar",start,end)["Close"]
    p_ticker = pd.DataFrame(p_ticker[t])
    p_ticker = p_ticker.rename(columns={"Close":t})
    p_monthly = p_ticker.resample(return_type).first()
    price_w = pd.concat([price_w,p_monthly],axis=1)
# print(price_w)

# 4. Data Processing

# 1) Derive the matrix of return variance
ret_all = pd.DataFrame()
for t in ticker:
    ret_all = pd.concat([ret_all,getret(t,price_w)],axis=1)
tmat_ret = ret_all.values.T
covmat = np.cov(tmat_ret)
# print(covmat)

# 2) Start portfolio optimization process

# i) Equal-Weight Method

weight = sp.ones(len(ticker)-1)/len(ticker) # start-point equal-weight combination
weight_equal = sp.append(weight,1-sum(weight))
std_equal = getvar(covmat,weight)
sharp_equal = getsharp(ret_all,riskfree,weight,covmat)

# ii) Min-Variance Method

w_minvar = fmin(testvar,weight) # length = len(ticker)-1
weight_minvar = sp.append(w_minvar,1-sum(w_minvar)) # length = len(ticker)
std_minvar = getvar(covmat,w_minvar)
sharp_minvar = getsharp(ret_all,riskfree,w_minvar,covmat)

# iii) Max-Sharp Method

w_maxsharp = fmin(testsharp,weight)
weight_maxsharp = sp.append(w_maxsharp,1-sum(w_maxsharp))
std_maxsharp = getvar(covmat,w_maxsharp)
sharp_maxsharp = getsharp(ret_all,riskfree,w_maxsharp,covmat)

# 5. Result Output

# Prepare base for output file
filename = "Portfolio Optimization.txt"
open(filename,"w").write("Portfolio Optimization Result\n")
print("Start Date:",start,file=open(filename,"a"))
print("End Date:",end,file=open(filename,"a"))

# Output the optimization result
print("===================================",file=open(filename,"a"))
print("Equal-Weight Optimize Portfolio",file=open(filename,"a"))
for i in range(len(ticker)):
    print("Weight of",ticker[i]+":",weight_equal[i],file=open(filename,"a"))
print("Portfolio Risk(Standard Deviation):",std_equal,file=open(filename,"a"))
print("Portfolio Sharp Ratio(Return per risk):",sharp_equal,file=open(filename,"a"))
print("===================================",file=open(filename,"a"))
print("Min-Variance Optimize Portfolio",file=open(filename,"a"))
for i in range(len(ticker)):
    print("Weight of",ticker[i]+":",weight_minvar[i],file=open(filename,"a"))
print("Portfolio Risk(Standard Deviation):",std_minvar,file=open(filename,"a"))
print("Portfolio Sharp Ratio(Return per risk):",sharp_minvar,file=open(filename,"a"))
print("===================================",file=open(filename,"a"))
print("Max-Sharp Optimize Portfolio",file=open(filename,"a"))
for i in range(len(ticker)):
    print("Weight of",ticker[i]+":",weight_maxsharp[i],file=open(filename,"a"))
print("Portfolio Risk(Standard Deviation):",std_maxsharp,file=open(filename,"a"))
print("Portfolio Sharp Ratio(Return per risk):",sharp_maxsharp,file=open(filename,"a"))
print("===================================",file=open(filename,"a"))
print("Optimize completed! Please check 'Portfolio Optimization.txt'")
