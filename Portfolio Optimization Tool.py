# Easy Tool for Portfolio Optimization
# Copyright----Ziqi Liu
# Input: Tickers / Start Date
# Output: HTML Report of Portfolio Optimization

import datetime as dt
import pandas as pd
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as pdr
from scipy.optimize import fmin

## 1. User Input

tick = input("Please input a ticker:")
ticker = []
while tick != "":
    ticker.append(tick)
    tick = input("Please input a ticker, press ENTER to stop:")

# Quit if tickers are not enough
if len(ticker) <= 1:
    print("Need more tickers input!")
    quit()

startyear = input("Please input the start year:")

# Quit if start date are not available
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

## 2. Function Definition

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
    ret = []
    for i in range(1,len(simret)):
        ret.append(simret[i])
    df_ret = pd.DataFrame(ret,columns=[ticker])
    return df_ret

# Function 3: portfolio variance calculation (actually, the return is standard deviation)
def getvar(covmat,weight): # length of weight = length of ticker - 1
    allweight = sp.append(weight,1-sum(weight))
    if all(w>=0 for w in allweight): # Use a boolean to ensure all weights > 0
        std_total = np.sqrt(np.dot(allweight.T,np.dot(covmat,allweight)))
    else:
        std_total = 9999
    return std_total

# Function 4: sharp ratio calculation
def getsharp(ret,rf,weight,covmat):
    allweight = sp.append(weight,1-sum(weight))
    portret = np.dot(allweight,ret.mean())
    std = getvar(covmat,weight)
    sharp = (portret-rf)/std
    return sharp

# Function 5: min variance function for fmin() optimization
# Be careful! outside variable: covmat
def findvar(weight): # length of weight = length of ticker - 1
    portvar = getvar(covmat,weight)
    return portvar

# Function 6: max sharp function for fmin() optimization
# Be careful! outside variables: ret_all, riskfree, covmat
def findsharp(weight):
    portsharp = -getsharp(ret_all,riskfree,weight,covmat)
    # Since it use fmin() to find the max sharp, the sharp should be a negative value
    return portsharp

## 3. Data Accessing

# Get monthly data from MorningStar
price_w = pd.DataFrame()
price_all = pd.DataFrame()
for t in ticker:
    p_ticker = pdr.DataReader(t,"morningstar",start,end)["Close"]
    p_ticker = pd.DataFrame(p_ticker[t])
    p_ticker = p_ticker.rename(columns={"Close":t})
    p_monthly = p_ticker.resample(return_type).first()
    price_w = pd.concat([price_w,p_monthly],axis=1)

## 4. Data Processing

# 1) Derive the matrix of return variance
ret_all = pd.DataFrame()
for t in ticker:
    ret_all = pd.concat([ret_all,getret(t,price_w)],axis=1)
covmat = np.cov(ret_all.values.T)

# 2) Start portfolio optimization process

# i) Equal-Weight Method

weight = sp.ones(len(ticker)-1)/len(ticker) # start-point equal-weight combination
weight_equal = sp.append(weight,1-sum(weight))
std_equal = getvar(covmat,weight)
sharp_equal = getsharp(ret_all,riskfree,weight,covmat)

# ii) Min-Variance Method

w_minvar = fmin(findvar,weight) # length = len(ticker)-1
weight_minvar = sp.append(w_minvar,1-sum(w_minvar)) # length = len(ticker)
std_minvar = getvar(covmat,w_minvar)
ret_minvar = np.dot(sp.append(w_minvar,1-sum(w_minvar)),ret_all.mean())
sharp_minvar = getsharp(ret_all,riskfree,w_minvar,covmat)

# iii) Max-Sharp Method

w_maxsharp = fmin(findsharp,weight)
weight_maxsharp = sp.append(w_maxsharp,1-sum(w_maxsharp))
std_maxsharp = getvar(covmat,w_maxsharp)
ret_maxsharp = np.dot(sp.append(w_maxsharp,1-sum(w_maxsharp)),ret_all.mean())
sharp_maxsharp = getsharp(ret_all,riskfree,w_maxsharp,covmat)

## 5. Efficient Frontier

# Generate random portfolios
num = 10000 # number of random portfolios
port_ret = []
port_vol = []
for oneport in range(num):
    randw = np.random.random(len(ticker))
    randw = randw/np.sum(randw) # sum the random weights to 1
    randret = np.dot(randw,ret_all.mean()) # mean monthly return
    randvol = getvar(covmat,randw[0:(len(ticker)-1)])
    port_ret.append(randret)
    port_vol.append(randvol)

portfolio = {"Returns":port_ret,"Risk":port_vol}
df_ef = pd.DataFrame(portfolio)

# Plot the efficient frontier
plt.style.use("ggplot")
fig_ef = df_ef.plot.scatter(x="Risk",y="Returns",figsize=(10,10),grid=True,alpha=0.3,s=2)
plt.xlabel("Portfolio Risk (Standard Deviation)")
plt.ylabel("Expected Returns")
plt.title("Efficient Frontier")
plt.scatter(x=std_maxsharp,y=ret_maxsharp,c="g",marker="*",s=200)
plt.annotate("Max-Sharp",(std_maxsharp,ret_maxsharp),color="g",size=12)
plt.scatter(x=std_minvar,y=ret_minvar,c="r",marker="*",s=200)
plt.annotate("Min-Var",(std_minvar,ret_minvar),color="r",size=12)
plt.savefig("Efficient Frontier.png")

## 6. Output html report

# Prepare the base for output file
filename = "Portfolio Optimization.html"
open(filename,"w").write("<html>\n")
print("<head><title>Portfolio Optimization Result</title></head>",file=open(filename,"a"))
print("<body>",file=open(filename,"a"))
print("<h1>Portfolio Optimization Report</h1>",file=open(filename,"a"))
print("<p>Start Date:",start,"</p>",file=open(filename,"a"))
print("<p>End Date:",end,"</p>",file=open(filename,"a"))

# Output the optimization result
print("<p>===================================</p>",file=open(filename,"a"))
print("<h2>Equal-Weight Portfolio</h2>",file=open(filename,"a"))
for i in range(len(ticker)):
    print("<p>Weight of",ticker[i]+":",weight_equal[i],"</p>",file=open(filename,"a"))
print("<p>Portfolio Risk(Standard Deviation):",std_equal,"</p>",file=open(filename,"a"))
print("<p>Portfolio Sharp Ratio(Return per risk):",sharp_equal,"</p>",file=open(filename,"a"))
print("<p>===================================</p>",file=open(filename,"a"))
print("<h2>Min-Variance Portfolio</h2>",file=open(filename,"a"))
for i in range(len(ticker)):
    print("<p>Weight of",ticker[i]+":",weight_minvar[i],"</p>",file=open(filename,"a"))
print("<p>Portfolio Risk(Standard Deviation):",std_minvar,"</p>",file=open(filename,"a"))
print("<p>Portfolio Sharp Ratio(Return per risk):",sharp_minvar,"</p>",file=open(filename,"a"))
print("<p>===================================</p>",file=open(filename,"a"))
print("<h2>Max-Sharp Portfolio</h2>",file=open(filename,"a"))
for i in range(len(ticker)):
    print("<p>Weight of",ticker[i]+":",weight_maxsharp[i],"</p>",file=open(filename,"a"))
print("<p>Portfolio Risk(Standard Deviation):",std_maxsharp,"</p>",file=open(filename,"a"))
print("<p>Portfolio Sharp Ratio(Return per risk):",sharp_maxsharp,"</p>",file=open(filename,"a"))
print("<p>===================================</p>",file=open(filename,"a"))

# Plot the efficient frontier
print("<h2>Efficient Frontier</h2>",file=open(filename,"a"))
print("<p><img src = 'Efficient Frontier.png'></p>",file=open(filename,"a"))

print("</body>",file=open(filename,"a"))
print("</html>",file=open(filename,"a"))
print("Optimize completed! Please check 'Portfolio Optimization.html'")
