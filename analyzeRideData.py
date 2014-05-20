import scipy.stats as st
import numpy as np
import pandas.io.parsers as parse
from pandas import DataFrame as df
import statsmodels.nonparametric.kernel_regression as kr
import statsmodels.tsa.arima_model as arma
import statsmodels.tsa.tsatools as tsa


#==================================================================================================#
#==================================================================================================#
#==================================================================================================#
class analyzeRideData:
	"""Class reads workout .csv files and processes / analyzes data"""
	#---------------------------------------------------
	def __init__(self, fileName, file_it):
		# Initialize variables
		self.fileName = fileName
		self.has_good_data = False
		self.file_it = file_it 							# file_it is useful for formatting results in driver

		# Read csv file
		self.ride_dataFrame = parse.read_csv(fileName)
		self.untrimmed_hr = []
		self.untrimmed_pw = []

		self.endog_var = []
		self.exog_var = []
		self.mle_prediction = []
		self.mle_param = None
		self.mle_resid = []
		self.mle_pos_resid = []


	#---------------------------------------------------
	#---------------------------------------------------
	#---------------------------------------------------
	def clean_ride_data(self):
		"""Function winsorizes / trims hrate and power data."""
		# Preliminary replacing / filling
		self.ride_dataFrame.Hrate.replace(-1,float('NaN'),inplace=True)
		self.ride_dataFrame.Hrate.replace(0,float('NaN'),inplace=True)
		self.ride_dataFrame.Watts.replace(float(-1),0,inplace=True)
		self.ride_dataFrame.Watts = self.ride_dataFrame.Watts.fillna(method='bfill')

		# Save hr data for visualization. Power data saved below, after winsorization
		self.untrimmed_hr = list(self.ride_dataFrame.Hrate)						# Important that this copy is deep...

		# Find ceiling, floor for valid data (probably not valid to assume gaussian distro.)
		power_ceiling = self.ride_dataFrame.Watts.quantile(.98)
		power_floor = max(self.ride_dataFrame.Watts.quantile(.25),20)
		hr_floor = self.ride_dataFrame.Hrate.fillna(method='bfill').fillna(method='ffill').quantile(.02)
		hr_ceiling = self.ride_dataFrame.Hrate.fillna(method='bfill').fillna(method='ffill').quantile(.98)

		# Winsorize power ceiling and trim power floor
		self.ride_dataFrame.Watts[self.ride_dataFrame.Watts > power_ceiling] = power_ceiling

		# Save power data for visualization. Hrate data saved above, before index shifting
		self.untrimmed_pw = list(self.ride_dataFrame.Watts)						# Important that this copy is deep...

		# Trim leading / lagging bad indices, reset index values		
		first_good_index = max(self.ride_dataFrame.Watts.first_valid_index(),
								self.ride_dataFrame.Hrate.first_valid_index())
		last_good_index = min(self.ride_dataFrame.Watts.last_valid_index(),
								self.ride_dataFrame.Hrate.last_valid_index())	
		if first_good_index is not None and last_good_index is not None and self.ride_dataFrame.Watts.max() > 0:
			self.has_good_data = True	
			self.ride_dataFrame = self.ride_dataFrame.ix[first_good_index:last_good_index]
			num_good_indices = last_good_index - first_good_index
			self.ride_dataFrame = self.ride_dataFrame.set_index(np.arange(0,num_good_indices+1))

		# Fill Hrate NaNs
		self.ride_dataFrame.Hrate.fillna(method='bfill').fillna(method='ffill')


	#---------------------------------------------------
	#---------------------------------------------------
	#---------------------------------------------------
	def get_bucket_hr_at(self,power,box_length=90):
		minutes_per_tick = self.ride_dataFrame.Minutes[1] - self.ride_dataFrame.Minutes[0]
		hr_boxes = [np.mean(self.ride_dataFrame.Hrate.shift(-23).fillna(method='bfill').fillna(method='ffill')[x:x+box_length]) for x in xrange(0, len(self.ride_dataFrame.Hrate), box_length)]
		pw_boxes = [np.mean(self.ride_dataFrame.Watts.fillna(method='bfill').fillna(method='ffill')[x:x+box_length]) for x in xrange(0, len(self.ride_dataFrame.Watts), box_length)]
		
		last_good_index = self.get_last_index_above(pw_boxes,1)
		hr_boxes = hr_boxes[:last_good_index]
		pw_boxes = pw_boxes[:last_good_index]

		power_hr_slope, intercept, r_value, p_value, std_err = st.linregress(pw_boxes,hr_boxes)
		hr_val = power*power_hr_slope + intercept
		return hr_val	


	#---------------------------------------------------
	#---------------------------------------------------
	#---------------------------------------------------
	def get_last_index_above(self,raw_list,min_acceptable):
		dist_from_back = len(raw_list)-1
		for i, val in enumerate(reversed(raw_list)):
			if val >= min_acceptable:
				dist_from_back = i
				break
		last_index = len(raw_list) - dist_from_back
		return last_index


	#---------------------------------------------------
	#---------------------------------------------------
	#---------------------------------------------------
	def get_first_index_above(self,raw_list,min_acceptable):
		dist_from_front = len(raw_list)-1
		for i, val in enumerate(raw_list):
			if val >= min_acceptable:
				dist_from_front = i
				break
		return dist_from_front


	#---------------------------------------------------
	#---------------------------------------------------
	#---------------------------------------------------
	def get_fitness_param(self):
		"""Function returns one scalar to indicate a rider's cardio fitness level based on ride data
			Define fitness by the average of two parameters
			param1 is the mean power / mean heartrate
			param2 is 1 / the hr at a given power determined by least squares regression of pw-hr distr
				multiplied by a scaling term to put param2 on the same order as param1
		"""
		fitness_param = 0

		if not self.has_good_data:
			self.clean_ride_data()
		if self.has_good_data:
			param1 = self.get_param1()
			param2 = self.get_param2()
			param3 = self.get_param3()
			param4 = self.get_param4()
			fitness_param = (param1 +  param2 + param3 + param4) / 4
		return fitness_param

	#---------------------------------------------------
	#---------------------------------------------------
	#---------------------------------------------------
	def get_param1(self):
		"""Function returns mean power / mean heartrate"""
		param1 = 0
		if self.has_good_data:
			mean_power = self.ride_dataFrame.Watts[self.ride_dataFrame.Watts>0].mean()
			mean_hrate = self.ride_dataFrame.Hrate[self.ride_dataFrame.Watts>0].mean()
			param1 = mean_power / mean_hrate
		return param1

	#---------------------------------------------------
	#---------------------------------------------------
	#---------------------------------------------------
	def get_param2(self):
		"""Function returns 1 / the hr at a given power determined
		 by least squares regression of pw-hr distr
		 """
		param2 = 0
		if self.has_good_data:
			token_power = 150		# 175 W seems to be at upper end of aerobic
			scale_hr_param = 150			
			param2 = scale_hr_param * (1.1 / self.get_bucket_hr_at(token_power))
		return param2		


	#---------------------------------------------------
	#---------------------------------------------------
	#---------------------------------------------------
	def get_param3(self):
		"""Function returns scale param / linear relationship between hr and power as determined
			by arima model.
		 """
		param3 = 0
		if self.has_good_data:
			scale_hr_param = 0.275			
			param3 = scale_hr_param * (1 / self.mle_param)
		return param3	

	#---------------------------------------------------
	#---------------------------------------------------
	#---------------------------------------------------
	def get_param4(self):
		"""Evaluates fitness as a measure of mean power divided by indicator of mean fatigue.
		The correlation between hr-power nonlinearity and fatigue is somewhat speculative given
		the fact that the rider is frequently in a transient metabolic/aerobic state. 
		 """
		param4 = 0
		if self.has_good_data:
			mean_pw = self.ride_dataFrame.Watts[self.ride_dataFrame.Watts>1].mean()
			error_list = self.mle_pos_resid				# Error corresponds to anaerobic resp.
			error_sum = np.sum(error_list)
			error_density = error_sum / len(error_list)		
			param4 = (mean_pw / error_density / 55)
		return param4	


	#---------------------------------------------------
	#---------------------------------------------------
	#---------------------------------------------------
	def get_hr_at(self, power):
		"""Function returns heartrate at a given power value for
		 ride data based on solving y=mx+b on regression parameters
		 """
		x = self.ride_dataFrame.Watts.fillna(method='ffill')
		y = self.ride_dataFrame.Hrate.fillna(method='bfill')
		power_hr_slope, intercept, r_value, p_value, std_err = st.linregress(x,y)
		hr_val = power*power_hr_slope + intercept
		return hr_val	


	#---------------------------------------------------
	#---------------------------------------------------
	#---------------------------------------------------
	def get_sum_pos_resid(self):
		return np.sum(self.mle_pos_resid)


	#---------------------------------------------------
	#---------------------------------------------------
	#---------------------------------------------------
	def perform_time_analysis(self):
		"""Function writes results for given ride object to pdf page"""
	
		print "Running MLE analysis for ",self.fileName

		minutes_per_tick = self.ride_dataFrame.Minutes[1] - self.ride_dataFrame.Minutes[0]
		minutes_per_box = 3
		box_size = int(minutes_per_box / minutes_per_tick)		# box_size < 1.5 min often produce non-stationary variables

		self.ride_dataFrame.Hrate = self.ride_dataFrame.Hrate.shift(-23).fillna(method='bfill').fillna(method='ffill')
		endog_var = self.get_box_list(list(self.ride_dataFrame.Hrate - self.ride_dataFrame.Hrate.mean()),box_size)
		exog_var = self.get_box_list(list(self.ride_dataFrame.Watts.fillna(method='ffill')),box_size)

		pw_floor = 20
		first_good_index = self.get_first_index_above(exog_var, 10)
		last_good_index = self.get_last_index_above(exog_var, pw_floor)
		if last_good_index > first_good_index+1 and np.max(exog_var) > 0:
			exog_var = exog_var[first_good_index:last_good_index]
			endog_var = endog_var[first_good_index:last_good_index]

			if len(exog_var):
				# endog_var = tsa.detrend(endog_var)
				p = 0
				q = 0
				d = 0
				model_order = (p,d,q)
				model = arma.ARIMA(endog_var,order=model_order,exog=exog_var)
				model_results = model.fit(method='css-mle')

				self.endog_var = endog_var
				self.exog_var = exog_var
				self.mle_prediction = model_results.fittedvalues
				self.mle_resid = model_results.resid

				self.mle_param = model_results.params[1]

				error_list = [0]*len(model_results.resid)
				for index, resid_val in enumerate(model_results.resid):
					if resid_val > 0:
						error_list[index] = resid_val
				self.mle_pos_resid = error_list


	#---------------------------------------------------
	#---------------------------------------------------
	#---------------------------------------------------
	def get_box_list(self, original_list, box_length):
		return [np.mean(original_list[x:x+box_length]) for x in xrange(0, len(original_list), box_length)]


