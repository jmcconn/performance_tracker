#!/usr/bin/python

import argparse
import pandas as pd
import numpy as np
import datetime as dt
import yfinance as yf

import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# add arguments for the trading activity and benchmark that should be loaded

parser = argparse.ArgumentParser()
parser.add_argument(dest='trading_activity_file', help='loads the desired trading activity', type=str)
parser.add_argument(dest='benchmark', help='loads the desired benchmark, can be SPY, ONEQ or IWF',type=str)
args = parser.parse_args()
trading_activity_file = args.trading_activity_file
benchmark = args.benchmark


# import csv file with trading activity

portfolio_activity = pd.read_csv('activity_files/' + trading_activity_file + '.csv', header=1)


# adjust date column to be datetime object

portfolio_activity['Date'] = pd.to_datetime(portfolio_activity['Date'])
portfolio_activity['Date'] = portfolio_activity['Date'].dt.date


# swap tickers that have changed

portfolio_activity.replace('FB', 'META', inplace=True)
portfolio_activity.replace('CR', 'CXT', inplace=True)


# load price and benchmark data

start_date = portfolio_activity.tail(1)['Date'].values[0]
end_date = portfolio_activity.head(1)['Date'].values[0]
all_tickers = portfolio_activity[portfolio_activity['Type'] == 'Investment']['Symbol'].unique()
all_tickers = [x for x in all_tickers if type(x) == str]
stock_price_data = yf.download(all_tickers, start=start_date, end=end_date)['Close']
benchmark_price_data = yf.download(['SPY','ONEQ','IWF'], start=start_date, end=end_date)['Close']
stock_price_data.replace(np.nan,0,inplace=True)


# adjust pricing data for stock splits

split_df = pd.DataFrame(columns=['Symbol','Split_Date','Split_Multiple'])

for ticker in all_tickers:
    split = yf.Ticker(ticker).splits
    split_dates = split.index.date
    split_multiple = split.values
    split_index = np.where((start_date <= split_dates) & (split_dates <= end_date))
    
    if len(split_index[0]) != 0:
        split_dates = split.index.date
        split_df.loc[len(split_df)] = [ticker, split_dates[split_index], int(split_multiple[split_index])]


for row in split_df.iterrows():
    pre_split = np.where(stock_price_data[row[1]['Symbol']].index.date < split_df[split_df['Symbol'] == row[1]['Symbol']]['Split_Date'].iloc[0][0])
    stock_price_data[row[1]['Symbol']].iloc[pre_split] = stock_price_data[row[1]['Symbol']].iloc[pre_split]*split_df[split_df['Symbol'] == row[1]['Symbol']]['Split_Multiple'].values[0]




# this function takes in trading activity and pricing data over a time period and builds a daily portfolio to determine a daily portfolio value for plotting to compare to benchmark

def update_portfolio(trading_activity, daily_pricing_data, benchmark_price_data):
    
    start_date = trading_activity.tail(1)['Date'].values[0]
    end_date = trading_activity.head(1)['Date'].values[0]
    date = start_date
    
    cash = 0
    transfer_amount = 0
    benchmark_shares = 0
    
    daily_portfolio = pd.DataFrame()
    portfolio_value_df = pd.DataFrame(columns=['Date', 'Portfolio_Value'])
    
    while date <= end_date:
        
	# skip non business days

        if date not in benchmark_price_data.index.date:
            date += dt.timedelta(days=1)
            continue
                
        else:
 		
	# calculate value of daily investment transfer, buy, sell activity
           
            investment_transfers = trading_activity[(trading_activity['Activity'].isin(['TRANSFER', 'RECEIVE DTC', 'RECEIVE'])) & (trading_activity['Date'] == date) & (trading_activity['Type'] == 'Investment')]
            
            investment_transfer_value = np.dot(investment_transfers['Quantity'].tolist(), 
                                               investment_transfers['Price'].tolist())
            
            investment_sale_value = trading_activity[(trading_activity['Activity'] == 'SOLD') & (trading_activity['Date'] == date) & (trading_activity['Symbol'] != None)]['Amount'].sum()
            
            investment_purchase_value = trading_activity[(trading_activity['Activity'] == 'BOUGHT') & (trading_activity['Date'] == date) & (trading_activity['Symbol'] != None)]['Amount'].sum()
            
	# calculate the number of benchmark shares that could be purchased for an equivalent amount of investment activity
            
            benchmark_shares += investment_transfer_value/benchmark_price_data[benchmark].loc[str(date)]
            benchmark_shares -= investment_purchase_value/benchmark_price_data[benchmark].loc[str(date)]
            benchmark_shares -= investment_sale_value/benchmark_price_data[benchmark].loc[str(date)]
            benchmark_value = benchmark_shares*benchmark_price_data[benchmark].loc[str(date)]
            
	# update portfolio on a daily basis  
            
            daily_stock_trades = trading_activity[(trading_activity['Date'] == date) 
                                                  & (trading_activity['Type'] == 'Investment')]
        
            
            daily_portfolio = pd.concat([daily_portfolio, daily_stock_trades]).groupby(['Symbol'], as_index=False).agg(
                                                    {'Account Number':'first','Date':'first','Activity':'first',
                                                    'Description':'first','Cusip':'first', 'Type':'first',
                                                    'Quantity':'sum', 'Price':'first', 'Amount':'first',
                                                    'Friendly Account Name':'first'})
	
	# drop all positions that have been closed out        

            daily_portfolio = daily_portfolio[daily_portfolio['Quantity'] != 0]
        
	# update daily changes to cash balance

            daily_cash_changes = trading_activity[(trading_activity['Date'] == date) 
                                                & (trading_activity['Type'] == 'Cash')]['Amount'].sum()
            
            daily_investment_changes = trading_activity[(trading_activity['Date'] == date) 
                                                      & (trading_activity['Type'] == 'Investment')]['Amount'].sum()
        
            total_daily_cash_changes = daily_cash_changes + daily_investment_changes
            cash += total_daily_cash_changes
        
        
            daily_prices = daily_pricing_data[daily_portfolio['Symbol'].values.tolist()].loc[dt.datetime.combine(date,dt.time(0,0,0))]  
            investment_value = np.dot(daily_prices.T.tolist(), daily_portfolio['Quantity'].tolist())
            portfolio_value = investment_value + cash
            
            total_benchmark_value = benchmark_value + cash
            
	# dataframe that includes daily portfolio and benchmark value
            
            portfolio_value_df = pd.concat([portfolio_value_df, pd.DataFrame({'Date':[date], 'Portfolio_Value':[portfolio_value], 'Benchmark_Value':[total_benchmark_value]})], ignore_index=True)
        
            date += dt.timedelta(days=1)
            
	# dataframe that includes the final portfolio at the end of date range    

    final_portfolio = daily_portfolio[['Symbol', 'Description', 'Quantity']]
    final_portfolio['Price'] = daily_prices.values
        
    return(portfolio_value_df)


print('Calculating daily portfolio value. This may take a few moments.')

portfolio_value = update_portfolio(portfolio_activity, stock_price_data, benchmark_price_data) 


# plot returns of portfolio and benchmark

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(portfolio_value['Date'], portfolio_value['Portfolio_Value'], label='portfolio')
ax.plot(portfolio_value['Date'], portfolio_value['Benchmark_Value'], label=benchmark)

ax.xaxis.set_major_locator(mdates.DayLocator(interval=90))
ax.tick_params(axis='x', labelrotation=45)
ax.set_xlabel('Date')

ax.yaxis.get_major_formatter().set_scientific(False)
ax.yaxis.get_major_formatter().set_useOffset(False)
ax.set_ylabel('Value ($)')

ax.legend()
plt.tight_layout()
plt.savefig('charts/' + trading_activity_file + ' vs. ' + benchmark + '.pdf')

print('Complete! Graph will appear in the charts directory.')
