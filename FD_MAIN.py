import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
import warnings
import pandas as pd
import time
import statsmodels.api as sm
os.chdir(os.getcwd()+"\Financial Derivatives\Coding")

DATA_daily = pd.read_csv('Boeing_stock_data.csv')
DATA_weekly = pd.read_csv('Boeing_stock_data_weekly.csv')
DATA_monthly = pd.read_csv('Boeing_stock_data_monthly.csv')

def Historical_stock():
    #Historial_stock_low.png
    
    date = pd.to_datetime(DATA_monthly.loc[:,'Date'])
    Adj_Close = DATA_monthly.loc[:,'Adj Close']
    
    plt.figure(figsize=(30,20))
    plt.rcParams.update({'font.size': 35})
    plt.rcParams.update({'lines.linewidth' : 4})
    #plt.rcParams.update({'figure.dpi' : 400})
    fig=plt.gcf()
    ax=fig.add_subplot(111)
    ax.plot(date, Adj_Close,color='black')
    ax.axvline(pd.to_datetime("2001-09-11"),lw=2,ls='--',color='black')#9/11
    ax.axvline(pd.to_datetime("2008-09-28"),lw=2,ls='--',color='black')#stock market crash
    ax.axvline(pd.to_datetime("2017-07-01"),lw=2,ls='--',color='black')#3rd divisoin open
    ax.axvline(pd.to_datetime("2019-03-10"),lw=2,ls='--',color='black')#Ethopian airlines crash
    ax.axvspan(xmin=pd.to_datetime("2019-03-10"),xmax=pd.to_datetime("2024-05-01"),alpha=0.35,color='black')#nvestigations    
    ax.set_ylabel('Adjusted Closing Price (USD $)')
    ax.set_xlabel('Year')
    ax.set_yticks([0,100,200,300,400,])
    ax.set_ylim(bottom=0)
    plt.grid()
    plt.subplots_adjust(bottom=0.113,right=0.99,top=0.99,left=0.1)
    #plt.savefig('Historial_stock_low.png',dpi=100)
    plt.show()

def Returns_HIST():
    G1 = True
    G2 = True
    G3 = True
    
    #1985 to 2024   
    if G1 == True:
        Open = (DATA_daily.loc[5778:,'Open']).to_numpy()    
        Close = (DATA_daily.loc[5778:,'Close']).to_numpy()
        D_Returns = (Close - Open) /  Open
        
        Open = (DATA_weekly.loc[1200:,'Open']).to_numpy()    
        Close = (DATA_weekly.loc[1200:,'Close']).to_numpy()#
        W_Returns = (Close - Open) /  Open
        
        Open = (DATA_monthly.loc[:,'Open']).to_numpy()    
        Close = (DATA_monthly.loc[:,'Close']).to_numpy()
        M_Returns = (Close - Open) /  Open
        
        ###PLOTTING###
        plt.figure(figsize=(30,20))
        plt.rcParams.update({'font.size': 35})
        plt.rcParams.update({'lines.linewidth' : 4})
        #plt.rcParams.update({'figure.dpi' : 400})
        fig=plt.gcf()
        ax=fig.add_subplot(111)
        ax.set_title('1985-2024')
        ax.hist(x=(D_Returns,W_Returns,M_Returns),label=('Daily','Weekly','Monthly'),bins=50,density=True,histtype='bar',)
        ax.set_ylabel('Probability Density')
        ax.set_xlabel('Daily Returns (Normalised)')
        ax.set_ylim(bottom=0)
        #ax.set_xlim(left=-0.5,right=0.7)
        plt.legend()
        plt.grid()
        plt.subplots_adjust(bottom=0.113,right=0.99,top=0.932,left=0.1)
        #plt.savefig('Historial_stock_low.png',dpi=100)
        plt.show()

    #2004 to 2024
    if G2 == True:
        Open = (DATA_daily.loc[10573:,'Open']).to_numpy()    
        Close = (DATA_daily.loc[10573:,'Close']).to_numpy()
        D_Returns = (Close - Open) /  Open
        
        Open = (DATA_weekly.loc[2192:,'Open']).to_numpy()    
        Close = (DATA_weekly.loc[2192:,'Close']).to_numpy()#
        W_Returns = (Close - Open) /  Open
        
        Open = (DATA_monthly.loc[228:,'Open']).to_numpy()    
        Close = (DATA_monthly.loc[228:,'Close']).to_numpy()
        M_Returns = (Close - Open) /  Open
        
        ###PLOTTING###
        plt.figure(figsize=(30,20))
        plt.rcParams.update({'font.size': 35})
        plt.rcParams.update({'lines.linewidth' : 4})
        #plt.rcParams.update({'figure.dpi' : 400})
        fig=plt.gcf()
        ax=fig.add_subplot(111)
        ax.set_title('2004-2024')
        ax.hist(x=(D_Returns,W_Returns,M_Returns),label=('Daily','Weekly','Monthly'),bins=50,density=True,histtype='bar',)
        ax.set_ylabel('Probability Density')
        ax.set_xlabel('Daily Returns (Normalised)')
        ax.set_ylim(bottom=0)
        #ax.set_xlim(left=-0.5,right=0.7)
        plt.legend()
        plt.grid()
        plt.subplots_adjust(bottom=0.113,right=0.99,top=0.932,left=0.1)
        #plt.savefig('Historial_stock_low.png',dpi=100)
        plt.show()
        
    #2023 to 024
    if G3 == True:
        Open = (DATA_daily.loc[15356:,'Open']).to_numpy()    
        Close = (DATA_daily.loc[15356:,'Close']).to_numpy()
        D_Returns = (Close - Open) /  Open
        
        Open = (DATA_weekly.loc[3183:,'Open']).to_numpy()    
        Close = (DATA_weekly.loc[3183:,'Close']).to_numpy()#
        W_Returns = (Close - Open) /  Open
        
        Open = (DATA_monthly.loc[456:,'Open']).to_numpy()    
        Close = (DATA_monthly.loc[456:,'Close']).to_numpy()
        M_Returns = (Close - Open) /  Open
        
        ###PLOTTING###
        plt.figure(figsize=(30,20))
        plt.rcParams.update({'font.size': 35})
        plt.rcParams.update({'lines.linewidth' : 4})
        #plt.rcParams.update({'figure.dpi' : 400})
        fig=plt.gcf()
        ax=fig.add_subplot(111)
        ax.set_title('2023-2024')
        ax.hist(x=(D_Returns,W_Returns,M_Returns),label=('Daily','Weekly','Monthly'),bins=50,density=True,histtype='bar',)
        ax.set_ylabel('Probability Density')
        ax.set_xlabel('Daily Returns (Normalised)')
        ax.set_ylim(bottom=0)
        #ax.set_xlim(left=-0.5,right=0.7)
        plt.legend()
        plt.grid()
        plt.subplots_adjust(bottom=0.113,right=0.99,top=0.932,left=0.1)
        #plt.savefig('Historial_stock_low.png',dpi=100)
        plt.show()
    
def Daily_returns_HIST():
    G1 = True
    G2 = True
    
    
    date = pd.to_datetime(DATA_daily.loc[:,'Date'])
    
    Open = (DATA_daily.loc[:,'Open']).to_numpy()    
    Close = (DATA_daily.loc[:,'Close']).to_numpy()
    diff = (Close - Open)
    D_Returns = diff /  Open
    
    #print(f"Standard Deviation: {np.std(D_Returns)}")
    #print(f"Mean: {np.mean(D_Returns)}")
    
    ###Normality tests###
    #print(stats.kstest(D_Returns,'norm'))
    #print(stats.anderson(x=diff,dist='norm'))
    print(stats.shapiro(D_Returns))
    
    ###PLOTTING###
    if G1 == True:
        plt.figure(figsize=(30,20))
        plt.rcParams.update({'font.size': 35})
        plt.rcParams.update({'lines.linewidth' : 4})
        #plt.rcParams.update({'figure.dpi' : 400})
        fig=plt.gcf()
        ax=fig.add_subplot(111)
        ax.hist(x=(D_Returns),bins=100,histtype='stepfilled',density=True,color='black',label='BA')
        ax.hist(np.random.normal(loc=np.mean(D_Returns),scale=np.std(D_Returns),size=int(1e7)),bins=int(1e3),histtype='step',density=True,color='red',label='Enhanced Gaussian ')
        ax.hist(np.random.normal(loc=np.mean(D_Returns),scale=np.std(D_Returns),size=len(D_Returns)),bins=100,histtype='step',density=True,color='blue',label='Reduced Gaussian')
        ax.set_ylabel('Probability Density')
        ax.set_xlabel('Daily Returns')
        ax.set_ylim(bottom=0.000)
        ax.set_xlim(left=-0.105,right=0.12)
        ax.set_xticks([-0.1,-0.05,0,0.05,0.1])
        plt.grid()
        plt.legend()
        plt.subplots_adjust(bottom=0.12,right=0.99,top=0.99,left=0.08)
        #plt.savefig('Historial_stock_low.png',dpi=100)
        plt.show()

    if G2 == True:
        ###Q-Q Plot###
        plt.figure(figsize=(30,20))
        plt.rcParams.update({'font.size': 35})
        plt.rcParams.update({'lines.linewidth' : 4})
        #plt.rcParams.update({'figure.dpi' : 400})
        fig=plt.gcf()
        ax=fig.add_subplot(111)
        sm.qqplot(D_Returns,ax=ax,line=('s'))
        ax.set_ylabel('Sample Quantiles')
        ax.set_xlabel('Theoretical Quantiles')
    #    ax.set_ylim(bottom=0.000)
    #    ax.set_xlim(left=-0.105,right=0.12)
        plt.grid()
        plt.legend()
        plt.subplots_adjust(bottom=0.12,right=0.99,top=0.99,left=0.1)
        #plt.savefig('Historial_stock_low.png',dpi=100)
        plt.show()

def Volatility():
    #3 Months: 15356
    #6 Months: 15293
    #9 Months: 15229
    #12 Months: 15167
    #2 Years: 14914
    #5 years 14409
    #10 Years: 13151
    #20 Years: 10637
    Index = 0
    Close = (DATA_daily.loc[15356:,'Close']).to_numpy()
    
    #Daily Returns#
    diff = np.zeros(len(Close-1))
    #Daily Drift #%
    for i in range(1,len(Close)):
        diff[i] = (Close[i]-Close[i-1])/Close[i-1]
    
    #Volatility
    D_Vol = np.std(diff)
    A_Vol = D_Vol *np.sqrt(252)
    
   
    #Drift
    D_Drift = (1/len(Close-1)) * np.sum(diff)
    A_drift = D_Drift * 252
    
    #print(f"Daily Volatility: {D_Vol*100} %")
    #print(f"Annual Volatility: {A_Vol*100} %")
    #print(f"Daily Drift: {D_Drift*100} %")
    #print(f"Annual Drift: {A_drift*100} %")
    
    Time = np.array(["0.25","0.50","0.75","1","2","5","10","20","Total"])
    VOL_ann = np.array([29.4,31.4,33.5,37.6,37.0,50.5,39.5,34.7,34.0])
    DRI_ann = np.array([-1.49,29.2,20.4,4.63,-4.25,-2.56,11.3,13.3,14.4])
    
    plt.figure(figsize=(30,20))
    plt.rcParams.update({'font.size': 35})
    plt.rcParams.update({'lines.linewidth' : 2})
    #plt.rcParams.update({'figure.dpi' : 400})
    fig=plt.gcf()
    ax=fig.add_subplot(111)
    #ax.plot(date_BA, BA_Hypo/(1000000*1.25),color='black',label='Boeing',ls='-')
    ax.plot(Time,VOL_ann,color='blue',label='Annualised Volatility',ls='--',markersize=10, marker='o')
    ax.plot(Time,DRI_ann,color='Red',label='Annualised Drift',ls='--',markersize=10, marker='o')
    ax.set_ylabel('%')
    ax.set_xlabel('Time period to Date (years)')
    #ax.set_yticks([0,100,200,300,400,])
    ax.set_ylim(bottom=-5,top=55)
    plt.legend()
    plt.grid()
    plt.subplots_adjust(bottom=0.12,right=0.99,top=0.97,left=0.08)
    #plt.savefig('Vol_Drift_HIGH.png',dpi=200,bbox_inches='tight')
    plt.show()

def Hypothetical():
    #Historial_stock_low.png
    G1 = True 
    #Boeing
    date_BA = pd.to_datetime(DATA_daily.loc[10573:,'Date'])
    Close = DATA_daily.loc[10573:,'Close'].to_numpy()
    Factor = (1000000*1.25)/Close[0]
    BA_Hypo = (Close * Factor)
    
    #Airbus
    Airbus_DATA = pd.read_csv('Airbus Daily.csv')
    date_Airbus = pd.to_datetime(Airbus_DATA.loc[:,'Date'])
    Close = Airbus_DATA.loc[:,'Close'].to_numpy()
    Factor = (1000000*1.25)/Close[0]
    Airbus_Hypo = (Close * Factor)

    #S&P 500
    SandP_DATA = pd.read_csv('S&P_500.csv')
    date_SandP = pd.to_datetime(SandP_DATA.loc[:,'Date'])

    Close = SandP_DATA.loc[:,'Close'].to_numpy()
    Factor = (1000000*1.25)/Close[0]
    SandP_Hypo = (Close * Factor)

    #LIBOR savings
    LIBOR_Data = pd.read_csv('historical-libor-rates-chart.csv')
    Rate = LIBOR_Data.loc[:,'Twelve'].to_numpy()
    Avg_rate = np.sum(Rate)/len(Rate)
    LIBOR_date = pd.to_datetime(['2004-01-01','2005-01-01','2006-01-01','2007-01-01','2008-01-01','2009-01-01','2010-01-01','2011-01-01','2012-01-01','2013-01-01','2014-01-01','2015-01-01','2016-01-01','2017-01-01','2018-01-01','2019-01-01','2020-01-01','2021-01-01','2022-01-01','2023-01-01','2024-01-01',])
    Savings = np.zeros(len(LIBOR_date))
    Savings[0] = 1000000*1.25
    for i in range(1,len(Savings)):
        Savings[i] = (Savings[i-1] * (Avg_rate/100)) + Savings[i-1]


    #OUTPUT info 
    METHOD = Savings
    print(f"Final Value: {METHOD[-1]}")
    print(f"Final Return: {100 * METHOD[-1]/(1000000*1.25)} %")
    print(F"Maximum Return: {100 * np.max(METHOD)/(1000000*1.25)} %")
    print(F"Minimum Return: {100 * np.min(METHOD)/(1000000*1.25)} %")
    
    if G1 == True:
        plt.figure(figsize=(30,20))
        plt.rcParams.update({'font.size': 35})
        plt.rcParams.update({'lines.linewidth' : 2})
        #plt.rcParams.update({'figure.dpi' : 400})
        fig=plt.gcf()
        ax=fig.add_subplot(111)
        ax.plot(date_BA, BA_Hypo/(1000000*1.25),color='black',label='Boeing',ls='-')
        ax.plot(date_Airbus, Airbus_Hypo/(1000000*1.25),color='Red',label='Airbus')
        ax.plot(date_SandP, SandP_Hypo/(1000000*1.25),color='blue',label='S&P500 Index')
        ax.plot(LIBOR_date, Savings/(1000000*1.25),color='Green',label=f'Savings account ({np.round(Avg_rate,2)}%)')
        ax.set_ylabel('Normalised Returns')
        ax.set_xlabel('Year')
        #x.set_xticks([0,100,200,300,400,])
        ax.set_ylim(bottom=0)
        plt.legend()
        plt.grid()
        plt.subplots_adjust(bottom=0.113,right=0.99,top=0.96,left=0.08)
        #plt.savefig('Historial_stock_low.png',dpi=100)
        plt.show()

########################################################################

#Function to obtain prices on a Binomial model
#As well as the Intrinsic option values
#We upload the maths library
import numpy as np
def Binomial(S,K,r,sigma,T,n,otype, EuOrAm, fair_price=False):
    #This function calculate the up, down ratio
    #and the risk-neutral probability
    #Inputs:
    #r: risk-free rate
    #sigma: volatility
    #T: Maturity time
    #n: number of steps
    # We calculate the time step sizez
    dt=T/n
    #We apply the u and d formulation
    u=np.exp(sigma*np.sqrt(dt))
    d=1/u
    #Then, we calculate the risk-neutral probability
    p=(np.exp(r*dt) -d)/(u-d)
    #We create a matrix to store the prices
    Stree=np.zeros((n+1, n+1))
    #We create a matrix to store the intrinic value
    Intrinsic=np.zeros((n+1, n+1))
    #We create a matrix to store the option value
    Option=np.zeros((n+1, n+1))
    #For each path
    for j in range(0,n+1):
        #For each time step
        for i in range(0,j+1):
            #The nodes are powers of u and d
            Stree[i,j] = S*(u**(j-i))*(d**(i))

            #Depending if is a call or a put
            #we apply a different payoff function
            if otype=="call":
                Intrinsic[i,j]=np.maximum(Stree[i,j]-K,0)
            elif otype=="put":
                Intrinsic[i,j]=np.maximum(K-Stree[i,j],0)
            else:
                print("Wrong option type. Please write call or put")
    #For each path
    for j in range(n,-1,-1):
        #For each time step
        for i in range(0, j+1):
            if j==n:
                Option[i,j] = Intrinsic[i,j]
            else:
                Option[i,j] = np.exp(-r*dt)*(Option[i,j+1]*p\
                                            + Option[i+1,j+1]*(1-p))
            #If it is American, we compare the option at the node
            #with its intrinsic value.
            #Choosing the maximum.
            if EuOrAm=="American":
                Option[i,j]=np.maximum(Intrinsic[i,j],Option[i,j])
    
    if fair_price == True:
        print("The Option price is", Option [0,0])

    return (u,d,p,Stree, Intrinsic, Option)
    

def Binomial_Actual():
    Binomial_table = False
    Binomial_graph = False
    Binomioal_function = True
    #Where the bread is made :)
    ############## Options (not Options) ###################################################    
    #Maturity Time
    T_mature=2/12 #Time measure in years, maturity time is 2 months i.e. 2/12 years
    
    #Steps in Binomial Model
    n=2
    
    #Option type
    otype="put"
    #otype="call"
    
    ############## VOLATILITY ###################################################
    Close = (DATA_daily.loc[15606:,'Close']).to_numpy()
    #Daily Returns#
    diff = np.zeros(len(Close-1))
    #Daily Drift #%
    for i in range(1,len(Close)):
        diff[i] = (Close[i]-Close[i-1])/Close[i-1]
    #Volatility
    D_Vol = np.std(diff)
    Vol = D_Vol * np.sqrt(252) #Annualised Volatility
    
    ############## Risk Free Rate - LIBOR #######################################
    LIBOR_Data = pd.read_csv('historical-libor-rates-chart.csv')
    Rate = LIBOR_Data.loc[:,'Twelve'].to_numpy()
    Avg_rate = (np.sum(Rate)/len(Rate))/100
    
    ############## Stock Pricings  ##############################################
    Start = (DATA_daily.loc[15672,'Close'])
    
    #Percentage factor to change     
    if otype == "put":
        Strike = Start * (0.90)
        
    elif otype == "call":
        Strike = Start * (1.10)
        
    else:
        print("ERROR")
        quit()

    #############################################################################
    if Binomial_table == True:
        print(Vol)
        OUT =  Binomial(Start,Strike,Avg_rate,Vol,T_mature,n,otype, "European",fair_price=True)
        for i in OUT:
            print(i)
        
    ### Bionomial Trees diagrams ###
    if Binomial_graph == True:
        OUT =  Binomial(Start,Strike,Avg_rate,Vol,T_mature,n,otype, "European")
        Stock_price = OUT[3]
        print(Stock_price)
        
        #TREE contains every possible path in tree hard coded causes im lazy 
        TREE = np.zeros(shape=[4,3])
        for i in range(4):
            TREE[i,0] = Stock_price[0,0]
            
        TREE[0,1] = Stock_price[0,1]
        TREE[1,1] = Stock_price[0,1]
        TREE[2,1] = Stock_price[1,1]
        TREE[3,1] = Stock_price[1,1]
        
        TREE[0,2] = Stock_price[0,2]
        TREE[1,2] = Stock_price[1,2]
        TREE[2,2] = Stock_price[1,2]
        TREE[3,2] = Stock_price[2,2]
        
        x = np.array(dtype=int,object=[0,1,2])
        
        plt.figure(figsize=(30,20))
        plt.rcParams.update({'font.size': 35})
        plt.rcParams.update({'lines.linewidth' : 2})
        plt.rcParams.update({'figure.dpi' : 200})
        
        fig=plt.gcf()
        ax=fig.add_subplot(111)
        #Plotting
        for i in range(4):
            ax.plot(x,TREE[i,:],color= 'black',marker='o',markersize=35,ls='--')
        
        #Node annotation
        ax.annotate(f'{TREE[0][0]:.2f}', (0, TREE[0][0]), textcoords="offset points", xytext=(0,25), ha='center')
        ax.annotate(f'{TREE[0][1]:.2f}', (1, TREE[0][1]), textcoords="offset points", xytext=(0,25), ha='center')
        ax.annotate(f'{TREE[2][1]:.2f}', (1, TREE[2][1]), textcoords="offset points", xytext=(0,25), ha='center')
        ax.annotate(f'{TREE[0][2]:.2f}', (2, TREE[0][2]), textcoords="offset points", xytext=(0,25), ha='center')
        ax.annotate(f'{TREE[1][2]:.2f}', (2, TREE[1][2]), textcoords="offset points", xytext=(0,25), ha='center')
        ax.annotate(f'{TREE[3][2]:.2f}', (2, TREE[3][2]), textcoords="offset points", xytext=(0,25), ha='center')
        
        #Node link annotation (p or 1-p)
        fontsize_link = 25
        p = OUT[2]
        
        ax.annotate(f'{(p):.2f}', (0, TREE[0][0]), textcoords="offset points", xytext=(260,57.5), ha='center',fontsize=fontsize_link,rotation=11)
        ax.annotate(f'{(1-p):.2f}', (0, TREE[0][0]), textcoords="offset points", xytext=(260,-51), ha='center',fontsize=fontsize_link,rotation=-11)
        
        ax.annotate(f'{(p):.2f}', (1, TREE[0][1]), textcoords="offset points", xytext=(260,62.5), ha='center',fontsize=fontsize_link,rotation=12)
        ax.annotate(f'{(1-p):.2f}', (1, TREE[0][1]), textcoords="offset points", xytext=(260,-57), ha='center',fontsize=fontsize_link,rotation=-11)
        
        ax.annotate(f'{(p):.2f}', (1, TREE[2][1]), textcoords="offset points", xytext=(260,50.5), ha='center',fontsize=fontsize_link,rotation=12)
        ax.annotate(f'{(1-p):.2f}', (1, TREE[2][1]), textcoords="offset points", xytext=(260,-47), ha='center',fontsize=fontsize_link,rotation=-10)
        
        ax.set_ylabel('Stock Price (USD $)')
        ax.set_xlabel('Time (Months)')
        ax.set_xticks(x)
        ax.set_xlim(left=-0.2,right=2.2)
        #ax.set_ylim(top=np.max(TREE)+20,bottom=np.min(TREE)-20)
        ax.set_ylim(top=240,bottom=140)
        plt.grid()
        plt.subplots_adjust(bottom=0.12,right=0.99,top=0.96,left=0.09)
        #plt.savefig('Historial_stock_low.png',dpi=100)
        plt.show()
 
    if Binomioal_function == True:
        ST = np.linspace(0,300,301)
        #ST = np.zeros(len(S))
        #PayOff = np.zeros(len(S))
        #PL = np.zeros(len(S))
        
        otype="put"
        #otype="call"
        if otype == "put":    
            Strike = Start * (1.08)
            print(f"Put Strike: {Strike:.2f}")
            
            OUT =  Binomial(Start,Strike,Avg_rate,Vol,T_mature,n,otype, "European",fair_price=True)
            Premimum = OUT[5]
            
            Long = np.maximum(Strike-ST, 0)
            PL = Long - Premimum[0,0]
            
        elif otype == "call":
            Strike = Start * (0.88)
            print(f"Call Strike: {Strike:.2f}")

            OUT =  Binomial(Start,Strike,Avg_rate,Vol,T_mature,n,otype, "European",fair_price=True)
            Premimum = OUT[5]
            
            Long = np.maximum(ST-Strike, 0)
            PL = Long - Premimum[0,0]
        else:
            print("ERROR")
            quit()

            
        plt.figure(figsize=(30,20))
        plt.rcParams.update({'font.size': 35})
        plt.rcParams.update({'lines.linewidth' : 4})
        plt.rcParams.update({'figure.dpi' : 200})
        fig=plt.gcf()
        ax=fig.add_subplot(111)
        ax.plot(ST, Long ,ls='-',color='black',label=f'Long {otype} payoff')
        ax.plot(ST, PL,ls='--',color='black',label=f'Long {otype} profit/loss')
        ax.set_ylabel('Returns (USD $)')
        ax.set_xlabel('Stock Price at Maturity (USD $)')
        ax.set_xlim(left=0,right=300)
        plt.legend()
        plt.grid()
        if otype == "put":
            plt.subplots_adjust(bottom=0.12,right=0.96,top=0.96,left=0.1)
        elif otype == "call":
            plt.subplots_adjust(bottom=0.12,right=0.96,top=0.96,left=0.1)
        #plt.savefig('Historial_stock_low.png',dpi=100)
        plt.show()
        
#Historical_stock()
#Returns_HIST()
#Daily_returns_HIST()
#Volatility()
#Hypothetical()
Binomial_Actual()