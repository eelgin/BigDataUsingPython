from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import urllib.request, json
import os
import numpy as np
import tensorflow as tf # This code has been tested with TensorFlow 1.6
from sklearn.preprocessing import MinMaxScaler
from PIL import Image

code_file = open('LSTM_WIP.py')
lines = code_file.readlines()

import inspect

def lineno():
    """Returns the current line number in our program."""
    return inspect.currentframe().f_back.f_lineno

import sys

# CMDline arg help printout
def print_help():
	print ("#########################################################################################")
	print ("FOR DEMO, PLEASE USE __")
	print ("NOTE: To use your own API key for AlphaVantage, paste it into the 'api_key' file")
	print ("#########################################################################################")
	print ("Usage \t*program name* -[k/a] --[kaggle/alphavantage] -st/--set_ticker *ticker* -h/--help")
	print ("#########################################################################################")
	print ("\t-h/--help || optional")
	print ("\t\tPrints this help message")
	print ("\t-k/-a || mandatory")
	print ("\t\tSelects Kaggle or AlphaVantage as datasource, respectively")
	print ("\t--kaggle/--alphavantage || mandatory alternative")
	print ("\t\tSelects Kaggle or AlphaVantage as datasource, respectively")
	print ("\t-st/--set_ticker || mandatory")
	print ("\t\tSelects stock ticker by initials ie. \'AAL\',\'INTC\',\'AAPL\'")
	print ("#########################################################################################")
	exit(1)

def process_args():
	# Datasource and Ticker placeholders
	ds = None
	ticker = None

	# CMDline arg processing
	# Default Values
	if (len(sys.argv) < 2):
		ds = 'kaggle'
		ticker = 'AAPL'
	else:
		set_ticker = False
		for arg in sys.argv:
			arg = arg.lower()
			if (arg == '-h') or (arg == '--help'):
				print_help()

			# TODO: check if valid ticker
			# Check to see if followup arg
			if (set_ticker == True) and (ticker == None):
				set_ticker = False
				ticker = arg

			# Check Args, two part args first
			if (arg == '-st') or (arg == '--set_ticker'):
				set_ticker = True
				continue

			# If the Source isnt set check
			if ds == None:
				if ((arg == '-k') or (arg == '--kaggle')):
					ds = 'kaggle'
				elif ((arg == '-a') or (arg == '--alphavantage')):
					ds = 'alphavantage'

	# final check
	if (ticker == None) or (ds == None):
		print ("ERROR: ")
		if ticker == None: print ("Ticker not set")
		if ds == None: print("Data Source not set")
		print_help()

	return ds, ticker

def print_code(lines):
	print('\t"""')
	for line in lines:
		line = line.replace("\t","",1) #remove first tab
		line = line.replace("\n","")
		print ("\t"+line)
	print ('\t"""')
	print ('')


def load_data(ds):
	data_source = ds

	#setup for piping out output
	print('\t"""')
	print('Loading Data...')

	print_code(lines[lineno():lineno()+50])
	print ("Loading " +  ticker + " data from " + data_source+ "...")
	if data_source == 'alphavantage':
	    # Loading Data from Alpha Vantage

	    # Load API Key from file
	    f = open("api_key", "r")
	    api_key = f.read()
	    if len(api_key) != 16:
	    	print ("Error: Invalid API Key")
	    	exit(3)
	    f.close()

	    # Ticker selected in kwargs processing
	    # ticker = ticker

	    # JSON file with all the stock market data for AAL from the last 20 years
	    url_string = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=%s&outputsize=full&apikey=%s"%(ticker,api_key)

	    # Setting up file to save to
	    file_to_save = 'stock_market_data-%s.csv'%ticker

	    # If files doesnt exit, download stock data, save data, and convert to pandas dataframe
	    if not os.path.exists(file_to_save):
	        with urllib.request.urlopen(url_string) as url:
	            data = json.loads(url.read().decode())
	            # extract stock market data
	            data = data['Time Series (Daily)']
	            df = pd.DataFrame(columns=['Date','Low','High','Close','Open'])
	            for k,v in data.items():
	                date = dt.datetime.strptime(k, '%Y-%m-%d')
	                data_row = [date.date(),float(v['3. low']),float(v['2. high']),
	                            float(v['4. close']),float(v['1. open'])]
	                df.loc[-1,:] = data_row
	                df.index = df.index + 1
	        print('Data saved to : %s'%file_to_save)
	        # Save file for future use        
	        df.to_csv(file_to_save)

	    # If file already created, load data from it
	    else:
	        print('File already exists. Loading data from CSV')
	        df = pd.read_csv(file_to_save)

	else:
	    # Load Data from pre-downloaded kaggle file of selected ticker
	    df = pd.read_csv(os.path.join('Stocks',ticker.lower()+'.us.txt'),delimiter=',',usecols=['Date','Open','High','Low','Close'])
	    print ('Opening kaggle CSV')

	# Sort DataFrame by date
	df = df.sort_values('Date')

	# Double check the result
	#print(df.head())
	#print(df.shape)

	return df

#import os
#os.system('cls' if os.name == 'nt' else 'clear')

import textwrap

def run_demo(df):
	# Because I cant get a python notebook to work I will be printing out all
	# relevant code lines after explanations
	
	input ('[Press enter to start the Demo]\n########################################################\n')

	print (textwrap.dedent("""\
			DEMO START:
	
			By this point several things have happened.
	
				1.a)	Stock Data has been downloaded then saved from the selected source
					or
				1.b) 	Stock Data has been loaded from a file from the selected source
				2.) 	Stock Data has been sorted by the date
	
			Lets take a quick peek at out data...
	
		"""))

	print_code(lines[lineno():lineno()+2])
	print (df.head())
	print (df.shape)
	input ('[Press enter to continue]\n########################################################\n')

	print (textwrap.dedent("""\
		As you can see, stock data has several numeric datapoints for any given day.
		These are...
			-Open: 	The stock price when the market opened for the day
			-Close:	The stock price when the market closed for the day
			-High:	The highest price at which a stock was traded for that day
			-Low:   The lowest price at which a stock was traded for that day
		While we could choose anyone of these 4 features to base our model off of,
		instead we are going to find the 'Mid' value of the day. This is done by
		simply averaging the High and the Low together. While this isn't a 'real'
		metric, by placing our estimates in the middle of the High and Low we 
		minimize the probability of getting positively or negatively biased price 
		predictions.
	"""))
	print_code(lines[lineno():lineno()+5])
	# First calculate the mid prices from the highest and lowest
	high_prices = df.loc[:,'High'].values
	low_prices = df.loc[:,'Low'].values
	mid_prices = (high_prices+low_prices)/2.0
	print (mid_prices[:6])
	input ('[Press enter to continue]\n########################################################\n')

	print (textwrap.dedent("""\
		Next we will split our data into train_data and test_data. Because the
		number of data points for each stock differs, we will calculate the index 
		for the cutoff data point that will separate the training and test data.
		When creating the data pool, its important that there is far more training
		data then testing data, so I will dedicate 85% of the data to training
		with the remaing 15% dedicated to testing.
	"""))
	print_code(lines[lineno():lineno()+6])
	cutoff = int(df.shape[0] * 0.85)
	print ("Cutoff: " + str(cutoff) + '/' + str(df.shape[0]))

	#setting training/testing data
	train_data = mid_prices[:cutoff]
	test_data = mid_prices[cutoff:] 
	input ('[Press enter to continue]\n########################################################\n')

	print (textwrap.dedent("""\
		Now that the Data is split up we can Normalize the data.
		Normalizing the data is essential for any algorithm to be able to 
		properly interpret and manipulate the data. Yet surprisingly, all 
		normailzing the data does is scales all of the data between 0 and 1.

		Why is this important?

		Well, if you were to normalize all of the data in one giant batch, nothing
		would be effected, the data would just be scaled down.

		However, by splitting up the normalization into batches, you can give the data
		and the changes between individual data points better context.

		To demonstrate this, lets take a peak at the stock ticker data.
		If you ran this program without any arguements the stock ticker chosen
		defaults to AAPL (Apple) data from Kaggle. This explanation will based upon that
		data.

		Take note of the shape of the stock data.
	"""))
	input ('[Press enter to open graph]\n########################################################\n')

	print_code(lines[lineno():lineno()+6])
	plt.figure(figsize = (18,9))
	plt.plot(range(df.shape[0]),(df['Low']+df['High'])/2.0)
	plt.xticks(range(0,df.shape[0],500),df['Date'].loc[::500],rotation=45)
	plt.xlabel('Date',fontsize=18)
	plt.ylabel('Mid Price',fontsize=18)
	plt.show(block=False)
	plt.savefig('fig1.png')



	print (textwrap.dedent("""\
		You may have noticed on the graph the tiny little bump up around 1999, and
		then the follow up little crash in late 2000. Let's take a peak at some dates
		from this timeperiod to see whats going on.
	"""))
	input ('[Press enter to continue]\n########################################################\n')

	print_code(lines[lineno():lineno()+1])
	print (df.iloc[[3740,3913,4058,4059,4080,4116]])
	print ('')
	print (textwrap.dedent("""\
		This small subset of that bump illustrates everything that happened. Over the
		course of 1999-2000, Apple's value had tripled from ~$1.35 to ~$4.05. Then,
		within the following 6 months Apple had lost almost 15% of its value before
		crashing the night of 9/28/2000. Closing that day Apple's value was at
		$3.42, but by opening its value had been sliced in half at $1.80. This downward
		trend would continue for 3 months before hitting rock bottom on 12/20/2000 at a
		value of $0.88.

		Looking at the graph zoomed out, you probubly just glanced over that bump as
		insignificant, a blip. However, that blip was a relatively major economic
		development and tradgedy for Apple at the time. This is because stock gains and
		losses are not measured in raw dollars and cents, but percentages and ratios.
		When a stocks value is in the thousands and it dips a dollar or two, there is
		no cause for alarm because that hit is less than a tenth of a percent of your
		investment. When Apple drops half of its value overnight, you wake up with half
		of your investments value gone into thin air.

		Normalizing the stock data over the whole dataset has this effect. The algorithm
		is looking at the data zoomed out. Those blips, no matter how important they were
		at the time gets growned out by the current magnitude of the stock price. But, by
		batching the normalization we are allowing the data to be put in context by having
		the magnitude of each value determined only by data within a certain time frame.

		Now lets run the batch normalization.
	"""))
	input ('[Press enter to continue]\n########################################################\n')

	print_code(lines[lineno():lineno()+22])
	#normalizing values, scales between 0 and 1
	scaler = MinMaxScaler()
	train_data = train_data.reshape(-1,1)
	test_data = test_data.reshape(-1,1)

	# Train the Scaler with training data and smooth data
	smoothing_window_size = 1000
	num_of_full_windows = int(cutoff / 1000)

	for di in range(0,num_of_full_windows*smoothing_window_size,smoothing_window_size):
		scaler.fit(train_data[di:di+smoothing_window_size,:])
		train_data[di:di+smoothing_window_size,:] = scaler.transform(train_data[di:di+smoothing_window_size,:])

	# Normalize the last bit of remaining data
	scaler.fit(train_data[di+smoothing_window_size:,:])
	train_data[di+smoothing_window_size:,:] = scaler.transform(train_data[di+smoothing_window_size:,:])

	# Reshape both train and test data
	train_data = train_data.reshape(-1)

	# Normalize test data
	test_data = scaler.transform(test_data).reshape(-1)

	print (textwrap.dedent("""\
		The test and training data is now normalized. To show this off lets take a
		quick peak at a few data points
	"""))

	input ('[Press enter to continue]\n########################################################\n')


	print_code(lines[lineno():lineno()+8])
	print (df.iloc[[4058]]),
	print (train_data[[4058]])
	print (df.iloc[[4059]]),
	print (train_data[[4059]])
	print (df.iloc[[6985]]),
	print (train_data[[6985]])
	print (df.iloc[[7067]]),
	print (train_data[[7067]])
	print('')

	print (textwrap.dedent("""\
		Here we can see just how effetive batch normalization is at applying context and
		scale to the data in a meaningful way. With the drop from $3.44 to $1.85 being 
		represented as 0.75 to 0.28, and a gain from $72.49 to $90.32 being represented
		as 0.87 to 0.98.

		Now before we can get onto different predictions methods we have to perfrom one
		last data preparation step. We will perform exponential moving average smoothing
		to just the training data.

		This will smooth over the day to day roughness of the stock data effectively
		removing the noise. This will allow the algorithms we use to more easily see 
		trends over longer periods.
	"""))

	input ('[Press enter to continue]\n########################################################\n')


	print_code(lines[lineno():lineno()+10])
	# Now perform exponential moving average smoothing
	# So the data will have a smoother curve than the original ragged data
	EMA = 0.0
	gamma = 0.1
	for ti in range(cutoff):
  		EMA = gamma*train_data[ti] + (1-gamma)*EMA
  		train_data[ti] = EMA

	# Used for visualization and test purposes
	all_mid_data = np.concatenate([train_data,test_data],axis=0)

	print (textwrap.dedent("""\
		With the data smoothed we can explore the simplist form of stock prediction,
		the Standard Average.

		The Standard Average predicts the value of a stock 1 day in the future by 
		averaging all prices observed within an arbitrarily specified window. 
		Put simply...

			The price at t + 1 is the average of all prices from t to t - N

		The implementation of such an algorithm is very simple...
	"""))
	
	input ('[Press enter to continue]\n########################################################\n')

	print_code(lines[lineno():lineno()+19])
	# Standard Average: One day ahead predictions
	window_size = 100
	N = train_data.size
	std_avg_predictions = []
	std_avg_x = []
	mse_errors = []

	for pred_idx in range(window_size,N):

	    if pred_idx >= N:
	        date = dt.datetime.strptime(k, '%Y-%m-%d').date() + dt.timedelta(days=1)
	    else:
	        date = df.loc[pred_idx,'Date']

	    std_avg_predictions.append(np.mean(train_data[pred_idx-window_size:pred_idx]))
	    mse_errors.append((std_avg_predictions[-1]-train_data[pred_idx])**2)
	    std_avg_x.append(date)

	print('MSE error for standard averaging: %.5f'%(0.5*np.mean(mse_errors)))
	print('')

	print (textwrap.dedent("""\
		Taking a look at the MSE error, we can see that this model is relatively accurate.
		The major downside though is that model can only accurately predict prices a single
		day in the future. This is not very useful, especially considering that this model 
		can't predict huge jump or dips in value.

		Let's take a peak at a comperative graph
	"""))

	input ('[Press enter to open graph]\n########################################################\n')
	
	print_code(lines[lineno():lineno()+9])
	plt.figure(figsize = (18,9))
	plt.plot(range(df.shape[0]),all_mid_data,color='b',label='True')
	plt.plot(range(window_size,N),std_avg_predictions,color='orange',label='Prediction')
	#plt.xticks(range(0,df.shape[0],50),df['Date'].loc[::50],rotation=45)
	plt.xlabel('Date')
	plt.ylabel('Mid Price')
	plt.legend(fontsize=18)
	plt.axvline(x=(cutoff-1))
	plt.show(block=False) 
	plt.savefig('fig2.png')
	
	print (textwrap.dedent("""\
		Next we will look at a more sophisticated alogrithm called Exponential Moving 
		Average (EMA). To calculate the prediction... (_of_ representing subscript)

			x_of_(t+1) = EMA_of_t = gamma * EMA_of_(t-1) + (1 - gamma) * x_of_t
			Where EMA_of_0 = 0

		The Exponential Moving Average is the Standard Average, but with an extra moving 
		part. The weight of the most recent prediction is decided by gamma. This scales 
		the impact of recent prices to be much more influential on predictions. This 
		fixes the issue of the algorithm not being able to predict or keep up with 
		sudden jumps and dips in stock value.

	"""))

	input ('[Press enter to continue\n########################################################\n')
	
	print_code(lines[lineno():lineno()+22])
	# Exponential Averaging
	window_size = 100
	N = train_data.size

	run_avg_predictions = []
	run_avg_x = []

	mse_errors = []

	running_mean = 0.0
	run_avg_predictions.append(running_mean)

	decay = 0.5

	for pred_idx in range(1,N):

	    running_mean = running_mean*decay + (1.0-decay)*train_data[pred_idx-1]
	    run_avg_predictions.append(running_mean)
	    mse_errors.append((run_avg_predictions[-1]-train_data[pred_idx])**2)
	    run_avg_x.append(date)

	print('MSE error for EMA averaging: %.5f'%(0.5*np.mean(mse_errors)))
	print ('')

	print (textwrap.dedent("""\
		As you can see the error for EMA is several magnitudes smaller than that of 
		Standard Averaging.

		Lets plot the EMA results.
	"""))

	input ('[Press enter view graph\n########################################################\n')
	
	print_code(lines[lineno():lineno()+8])
	plt.figure(figsize = (18,9))
	plt.plot(range(df.shape[0]),all_mid_data,color='b',label='True')
	plt.plot(range(0,N),run_avg_predictions,color='orange', label='Prediction')
	#plt.xticks(range(0,df.shape[0],50),df['Date'].loc[::50],rotation=45)
	plt.xlabel('Date')
	plt.ylabel('Mid Price')
	plt.legend(fontsize=18)
	plt.show()
	plt.savefig('fig3.png')

	print (textwrap.dedent("""\
		Zoomed out you might not have been able to tell that the 'True' prices for
		the training data was even graphed. It was, the model was just so accurate
		that while zoomed out the two graphs appear to be completely overlapped.

		While this model does seem impressive at being able to calculate a near
		exact copy of the model, it can only calculate a given stock value one
		day in advance which is not very useful. In fact, the way the algorithm is
		setup, trying to predict values any further than a single step (a day
		in this case) gives you the same answer for all future predictions.

		What would be useful is a model that can predict any arbitrary number of steps
		into the future. Luckily there does exist a model that can do this...

		The Long Short-Term Memory Model (LSTM)
	"""))

	input ('[Press enter to continue\n########################################################\n')
	
	print (textwrap.dedent("""\
		Long Short-Term Memory Models are extreamly powerful at making predictions
		based upon a time-series and can predict any number of steps into the future.
		The exact accuracy and power of a given LSTM model though is very dependent 
		on its fine-tuning for a given application, thus falling heavily on the 
		programmer implementing it.

		But before we get to all of that, lets break down what exactly is an LSTM.
	"""))

	input ('[Press enter to continue\n########################################################\n')
	
	# TODO: have NN_example print here

	print (textwrap.dedent("""\
		An LSTM is a type of Neural Network (NN), a Recurrent Neural Network(RNN)
		to be exact. NN are are effectively machine brains that are designed to
		complete one abstract task that conventional programming cannot solve.

		A common example is identifying handwritten numbers. 

		While it sounds simple even some of the best NNs only have an accuracy rate 
		of 98%. And at a huge computational overhead. This is because neural networks 
		are a network of connections between layers that contain thousands upon thousands 
		of nodes. These nodes are either an input, output, or some simple arithmatic 
		expression. The power of NN are breaking down complex tasks into a network of 
		simple arithmetic expressions. The output of these expressions are the weighted 
		by a bias that is calulated when the NN is trained. These biases and training
		are an optimization problem that incrementaly makes the NN more accurate.

		A NN has the issue of forgetting its output as soon as it finishes its 
		calulations. It can't use previously crunched data to influence it performance
		outside of the training process. While this isn't an issue for identifying 
		handwritten numbers, it would be for predicting stock prices. You can't train
		an NN with live data to have it adapt to new information unless you design it
		that way, which is exactly what a Recurrent Neural Network is.
	"""))
 	
	input ('[Press enter to continue\n########################################################\n')


	# TODO insert RNN_ref here

	print (textwrap.dedent("""\
 		RNN are designed with a memory of sorts that allow it to adjust its performance
 		based on data recieved during testing. This is essential for predicting stock
 		data especially if you want your predictions to be able to update without
 		retraining the whole system.

 		That is why LSTMs are a type of RNN.
 	"""))

	input ('[Press enter to continue\n########################################################\n')

	# TODO insert LSTM_diag here

	print (textwrap.dedent("""\
		An LSTM is comprised of 5 components:
			*Cell State 	= The internal memory of a cell for both the short and long
								term memories.
			*Hidden State 	= The output state calculated from the current input,
								current cell input, and the previous hidden state.
								This is used to predict the future stock prices.
								Can decide to retrive either both short and long 
								term memory or just one of the two.
			*Input Gate 	= Determines the amount of information from the input state
								that flows into the current cell state.
			*Forget Gate	= Determines the amount of information from the input state
								and previous cell state flows into the current cell 
								state.
			*Output Gate 	= Determines how much information from the current state
								flows into the hidden state

		These 5 components form modules that an LSTM is comprised of. While these 
		technical descriptions each component are useful at understanding what each
		part is responsible for, it understates the ultimate goal of these components
		within the module.

	"""))

	input ('[Press enter to continue\n########################################################\n')

	# TODO insert LSTM_CORE_DIAG.png here

	print (textwrap.dedent("""\
		The Cell state as shown in the image is the bottom line. The best explaination
		I've found for the Cell State is that it's like a converyor belt. 

		"It runs straight down the entire chain, with only some minor linear interactions. 
		 It’s very easy for information to just flow along it unchanged."

		The LSTM has the ability to add and remove information from this conveyor belt,
		thats where the gates come into play.
	"""))

	input ('[Press enter to continue\n########################################################\n')

	print (textwrap.dedent("""\
		Let us get to building a Long Short-Term Memory Model. The first thing we have 
		to do though is more data formatting.

		We are going to implement a 'data generator' to train our model. The core 
		component of the 'data generator' is it's 'unroll_batches()' method.
	"""))

	print_code(lines[lineno():lineno()+1])
	print (textwrap.dedent("""\
		
	"""))

def main_menu(df):
	print ("Enter an option below:")
	print ("'Demo'\tFull Demo of Project")
	print ("1.)\tPrint dataframe head")
	print ("2.)\tGraph dataframe")
	print ("Q.)\tGraph dataframe")

	while 1:
		choice = input(">")

		if choice.lower() == 'demo':
			return choice.lower()

		if choice.lower() == 'q' or choice.lower() == 'quit':
			code_file.close()
			exit(0)

		try:
			choice = int(choice)
		except Exception as e:
			print ("Invalid input")
			continue

		if (choice > 0) and (choice < 3):
			return choice
		else:
			print ("Invalid number")

	return int(input(">"))

if __name__ == '__main__':
	ds, ticker = process_args()
	df = load_data(ds)
	run_demo(df)
	"""
	while 1:
		choice = main_menu(df)

		if choice == 'demo':
			run_demo(df)
		elif choice == 1:
			print(df.head())
			print(df.shape)
		elif choice == 2:
			plt.figure(figsize = (18,9))
			plt.plot(range(df.shape[0]),(df['Low']+df['High'])/2.0)
			plt.xticks(range(0,df.shape[0],500),df['Date'].loc[::500],rotation=45)
			plt.xlabel('Date',fontsize=18)
			plt.ylabel('Mid Price',fontsize=18)
			plt.show()
	"""
	code_file.close()