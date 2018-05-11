# Easy Tool for Portfolio Optimization
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

print("Easy Tool for Portfolio Optimization")
print("===================================")
print("Please input stock ticker, press ENTER to stop(e.g. AAPL)")
tick = input("Ticker[1]:")
ticker = []
tickcount = 2
while tick != "":
    ticker.append(tick)
    tick = input("Ticker[{}]:".format(tickcount))
    tickcount += 1

# Quit if tickers are not enough
if len(ticker) <= 1:
    print("Need more tickers input!")
    quit()

print("Please input the year of start date(e.g. 2017)")
startyear = input("Start Year:")

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
riskfree = 0.01 # Set a risk free interest rate

## 2. Function Definition

# Function 1: accessing prices data
def getprice(ticker,start,end):
    data = pdr.DataReader(ticker,"morningstar",start,end)
    price = data["Close"] # Use closed price
    p = []
    for i in range(1,len(price)):
        p.append(price[i])
    df_price = pd.DataFrame(p,columns=[ticker])
    return df_price

# Function 2: accessing returns data
def getret(ticker,df_price):
    price = df_price[ticker]
    simret = price.pct_change(1) # Simple returns = percentage change of prices
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

# Function 7: generate a list of dates with one-month intervals
def datelist(begdate,enddate):
    datelist = []
    for date in pd.date_range(start=begdate,end=enddate,freq="m"):
        datelist.append(dt.datetime.strftime(date,"%y-%m"))
    return datelist

## 3. Data Accessing

# Get monthly data from MorningStar
price_w = pd.DataFrame()
price_all = pd.DataFrame()
for t in ticker:
    p_ticker = pdr.DataReader(t,"morningstar",start,end)["Close"]
    p_ticker = pd.DataFrame(p_ticker[t]) # Convert to data frame
    p_ticker = p_ticker.rename(columns={"Close":t}) # Rename columns to the name of tickers
    p_monthly = p_ticker.resample(return_type).first() # Resample to monthly prices
    price_w = pd.concat([price_w,p_monthly],axis=1)

## 4. Data Processing

# 1) Derive the matrix of return variance
ret_all = pd.DataFrame()
for t in ticker:
    ret_all = pd.concat([ret_all,getret(t,price_w)],axis=1)
covmat = np.cov(ret_all.values.T)

# 2) Start portfolio optimization process

# i) Equal-Weight Method

weight = sp.ones(len(ticker)-1)/len(ticker)
weight_equal = sp.append(weight,1-sum(weight))
std_equal = getvar(covmat,weight)
ret_equal = np.dot(weight_equal,ret_all.mean())
sharp_equal = getsharp(ret_all,riskfree,weight,covmat)

# ii) Min-Variance Method

w_minvar = fmin(findvar,weight) # length of w_minvar = len(ticker)-1
weight_minvar = sp.append(w_minvar,1-sum(w_minvar)) # length of weight_minvar = len(ticker)
std_minvar = getvar(covmat,w_minvar)
ret_minvar = np.dot(sp.append(w_minvar,1-sum(w_minvar)),ret_all.mean())
sharp_minvar = getsharp(ret_all,riskfree,w_minvar,covmat)

# iii) Max-Sharp Method

w_maxsharp = fmin(findsharp,weight)
weight_maxsharp = sp.append(w_maxsharp,1-sum(w_maxsharp))
std_maxsharp = getvar(covmat,w_maxsharp)
ret_maxsharp = np.dot(sp.append(w_maxsharp,1-sum(w_maxsharp)),ret_all.mean())
sharp_maxsharp = getsharp(ret_all,riskfree,w_maxsharp,covmat)

print("Report generating...")

## 5. Efficient Frontier

# Generate random portfolios
num = 10000*len(ticker) # Number of random portfolios
port_ret = []
port_vol = []
for oneport in range(num):
    randw = np.random.random(len(ticker))
    randw = randw/np.sum(randw) # Sum the random weights to 1
    randret = np.dot(randw,ret_all.mean()) # Mean of monthly return
    randvol = getvar(covmat,randw[0:(len(ticker)-1)])
    port_ret.append(randret)
    port_vol.append(randvol)

# Create a data frame of returns and risks
portfolio = {"Returns":port_ret,"Risk":port_vol}
df_ef = pd.DataFrame(portfolio)

# Plot the efficient frontier
plt.style.use("seaborn")
fig_ef = df_ef.plot.scatter(x="Risk",y="Returns",figsize=(10,10),alpha=0.5,s=2,c="tab:blue")
plt.xlabel("Portfolio Risk (Standard Deviation)")
plt.ylabel("Expected Returns")
plt.title("Efficient Frontier")
plt.scatter(x=std_equal,y=ret_equal,label="Equal",c="tab:purple",edgecolors="w",linewidths=1.5,marker="*",s=300)
plt.scatter(x=std_minvar,y=ret_minvar,label="MinVar",c="tab:green",edgecolors="w",linewidths=1.5,marker="*",s=300)
plt.scatter(x=std_maxsharp,y=ret_maxsharp,label="MaxSharp",c="tab:red",edgecolors="w",linewidths=1.5,marker="*",s=300)
plt.legend(frameon=True,loc=2)
plt.savefig("Efficient Frontier.png")
plt.close()

# 6. Back-test Analysis

# i) Equal-Weight Method
btret_equal = np.dot(weight_equal,ret_all.T) # Calculate historical monthly returns
btv_equal = 100 # Set the start value of back-test portfolio
btvlist_equal = [] # Create a base list of back-test value
for r in btret_equal:
    btvlist_equal.append(btv_equal)
    btv_equal = btv_equal*(1+r)

# ii) Min-Variance Method
btret_minvar = np.dot(weight_minvar,ret_all.T)
btv_minvar = 100
btvlist_minvar = []
for r in btret_minvar:
    btvlist_minvar.append(btv_minvar)
    btv_minvar = btv_minvar*(1+r)

# iii) Max-Sharp Method
btret_maxsharp = np.dot(weight_maxsharp,ret_all.T)
btv_maxsharp = 100
btvlist_maxsharp = []
for r in btret_maxsharp:
    btvlist_maxsharp.append(btv_maxsharp)
    btv_maxsharp = btv_maxsharp*(1+r)

# Create a datelist
date_list = datelist(startyear+"0101","20180101")

# Combine all lists to one dataframe
df_btvalue = pd.DataFrame({"Date":date_list,"Equal":btvlist_equal,"MinVar":btvlist_minvar,"MaxSharp":btvlist_maxsharp})

# Plot the back-test return
plt.style.use("seaborn")
plt.figure(figsize=(10,10))
fig_bt = plt.plot(df_btvalue["Date"],df_btvalue["Equal"],c="tab:purple")
plt.plot(df_btvalue["Date"],df_btvalue["MinVar"],c="tab:green")
plt.plot(df_btvalue["Date"],df_btvalue["MaxSharp"],c="tab:red")
plt.xticks(df_btvalue["Date"],rotation=60)
plt.xlabel("Date (Year-month)")
plt.ylabel("Portfolio Value Growth")
plt.title("Back-test Analysis")
plt.legend(frameon=True,loc=2)
plt.savefig("Back-test Analysis.png")
plt.close()

## 7. Output HTML Report

# Prepare the base for output file
filename = "Portfolio Optimization.html"
open(filename,"w").write("<html>\n")
print("<head><title>Portfolio Optimization Result</title></head>",file=open(filename,"a"))
print("<body>",file=open(filename,"a"))
print("<h1>Portfolio Optimization Report</h1>",file=open(filename,"a"))
print("<p>Ticker:",ticker,"</p>",file=open(filename,"a"))
print("<p>Start Date:",start,"</p>",file=open(filename,"a"))
print("<p>End Date:",end,"</p>",file=open(filename,"a"))

# Output the optimization result
print("<p>===================================</p>",file=open(filename,"a"))
print("<h2>Equal-Weight Portfolio</h2>",file=open(filename,"a"))
for i in range(len(ticker)):
    print("<p>Weight of ["+ticker[i]+"]:",weight_equal[i],"</p>",file=open(filename,"a"))
print("<p>Portfolio Risk(Standard Deviation):",std_equal,"</p>",file=open(filename,"a"))
print("<p>Portfolio Sharp Ratio(Return per risk):",sharp_equal,"</p>",file=open(filename,"a"))
print("<p>===================================</p>",file=open(filename,"a"))
print("<h2>Min-Variance Portfolio</h2>",file=open(filename,"a"))
for i in range(len(ticker)):
    print("<p>Weight of ["+ticker[i]+"]:",weight_minvar[i],"</p>",file=open(filename,"a"))
print("<p>Portfolio Risk(Standard Deviation):",std_minvar,"</p>",file=open(filename,"a"))
print("<p>Portfolio Sharp Ratio(Return per risk):",sharp_minvar,"</p>",file=open(filename,"a"))
print("<p>===================================</p>",file=open(filename,"a"))
print("<h2>Max-Sharp Portfolio</h2>",file=open(filename,"a"))
for i in range(len(ticker)):
    print("<p>Weight of ["+ticker[i]+"]:",weight_maxsharp[i],"</p>",file=open(filename,"a"))
print("<p>Portfolio Risk(Standard Deviation):",std_maxsharp,"</p>",file=open(filename,"a"))
print("<p>Portfolio Sharp Ratio(Return per risk):",sharp_maxsharp,"</p>",file=open(filename,"a"))
print("<p>===================================</p>",file=open(filename,"a"))

# Plot the efficient frontier
print("<h2>Efficient Frontier</h2>",file=open(filename,"a"))
print("<p>Plot Process:</p>",file=open(filename,"a"))
print("<p>1. Plot 10000 portfolios of random weight combinations</p>",file=open(filename,"a"))
print("<p>2. Mark the points of Min-Variance and Max-Sharp portfolios</p>",file=open(filename,"a"))
print("<p><img src = 'Efficient Frontier.png'></p>",file=open(filename,"a"))
print("<p>===================================</p>",file=open(filename,"a"))

# Plot the back-test analysis
print("<h2>Back-test Analysis</h2>",file=open(filename,"a"))
print("<p>Analysis Process:</p>",file=open(filename,"a"))
print("<p>1. Construct portfolios with the start value of $100</p>",file=open(filename,"a"))
print("<p>2. Simulate the portfolio returns based on historical data</p>",file=open(filename,"a"))
print("<p><img src = 'Back-test Analysis.png'></p>",file=open(filename,"a"))

# Close file
print("</body>",file=open(filename,"a"))
print("</html>",file=open(filename,"a"))
print("Report completed! Please check 'Portfolio Optimization.html'")
