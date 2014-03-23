import scipy.stats as st
import matplotlib.pyplot as plt
import sys, getopt, os, math, glob, numpy
import pandas.io.parsers as parse
from pandas import DataFrame as df
import statsmodels.nonparametric.kernel_regression as kr
import statsmodels.tsa.arima_model as arma
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.tight_layout as plt_l


#==================================================================================================#
#==================================================================================================#
#==================================================================================================#
class rideData:
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

		# Shift hr values. This accomodates for lag in hrate responses to power impulses and improves quality of hrate-power distribution
		self.ride_dataFrame.Hrate = self.ride_dataFrame.Hrate.shift(-23)

		# Fill NaN data points with closest valid point to left of NaN
		self.ride_dataFrame.Hrate = self.ride_dataFrame.Hrate.fillna(method='bfill')
		self.ride_dataFrame.Hrate = self.ride_dataFrame.Hrate.fillna(method='ffill')

		# Find ceiling, floor for valid data (probably not valid to assume gaussian distro.)
		power_ceiling = self.ride_dataFrame.Watts.quantile(.98)
		power_floor = max(self.ride_dataFrame.Watts.quantile(.25),20)
		hr_floor = self.ride_dataFrame.Hrate.quantile(.02)
		hr_ceiling = self.ride_dataFrame.Hrate.quantile(.98)

		# Winsorize power ceiling and trim power floor
		self.ride_dataFrame.Watts[self.ride_dataFrame.Watts > power_ceiling] = power_ceiling

		# Save power data for visualization. Hrate data saved above, before index shifting
		self.untrimmed_pw = list(self.ride_dataFrame.Watts)						# Important that this copy is deep...

		# Trim power floor
		# self.ride_dataFrame.Watts = self.ride_dataFrame.Watts[self.ride_dataFrame.Watts > power_floor]
		# self.ride_dataFrame.Hrate = self.ride_dataFrame.Hrate[self.ride_dataFrame.Watts > power_floor]

		# Trim hr floor, hr ceiling
		self.ride_dataFrame.Watts = self.ride_dataFrame.Watts[self.ride_dataFrame.Hrate > hr_floor]
		self.ride_dataFrame.Watts = self.ride_dataFrame.Watts[self.ride_dataFrame.Hrate < hr_ceiling]
		self.ride_dataFrame.Hrate = self.ride_dataFrame.Hrate[self.ride_dataFrame.Hrate > hr_floor]
		self.ride_dataFrame.Hrate = self.ride_dataFrame.Hrate[self.ride_dataFrame.Hrate < hr_ceiling]

		# Trim leading / lagging bad indices, reset index values		
		first_good_index = max(self.ride_dataFrame.Watts.first_valid_index(),
								self.ride_dataFrame.Hrate.first_valid_index())
		last_good_index = min(self.ride_dataFrame.Watts.last_valid_index(),
								self.ride_dataFrame.Hrate.last_valid_index())	
		if first_good_index is not None and last_good_index is not None and self.ride_dataFrame.Watts.max() > 0:
			self.has_good_data = True	
			self.ride_dataFrame = self.ride_dataFrame.ix[first_good_index:last_good_index]
			num_good_indices = last_good_index - first_good_index
			self.ride_dataFrame = self.ride_dataFrame.set_index(numpy.arange(0,num_good_indices+1))


	def get_bucket_hr_at(self,power):
		box_length = 90
		minutes_per_tick = self.ride_dataFrame.Minutes[1] - self.ride_dataFrame.Minutes[0]
		hr_boxes = [numpy.mean(self.ride_dataFrame.Hrate.shift(0).fillna(method='bfill').fillna(method='ffill')[x:x+box_length]) for x in xrange(0, len(self.ride_dataFrame.Hrate), box_length)]
		pw_boxes = [numpy.mean(self.ride_dataFrame.Watts.fillna(method='bfill').fillna(method='ffill')[x:x+box_length]) for x in xrange(0, len(self.ride_dataFrame.Watts), box_length)]
		
		last_good_index = self.get_last_index_above(pw_boxes,1)
		hr_boxes = hr_boxes[:last_good_index]
		pw_boxes = pw_boxes[:last_good_index]
		print last_good_index

		power_hr_slope, intercept, r_value, p_value, std_err = st.linregress(pw_boxes,hr_boxes)
		hr_val = power*power_hr_slope + intercept
		return hr_val	

	def get_last_index_above(self,raw_list,min_acceptable):
		dist_from_back = len(raw_list)
		for i, val in enumerate(reversed(raw_list)):
			if val >= min_acceptable:
				dist_from_back = i
				break
		last_index = len(raw_list) - dist_from_back
		return last_index


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
			fitness_param = (param1 + param2) / 2
		return param2

	#---------------------------------------------------
	def get_param1(self):
		"""Function returns mean power / mean heartrate"""
		param1 = 0
		if self.has_good_data:
			mean_power = self.ride_dataFrame.Watts.mean()
			mean_hrate = self.ride_dataFrame.Hrate.mean()
			param1 = mean_power / mean_hrate
		return param1

	#---------------------------------------------------
	def get_param2(self):
		"""Function returns 1 / the hr at a given power determined
		 by least squares regression of pw-hr distr
		 """
		param2 = 0
		if self.has_good_data:
			token_power = 50		# 175 W seems to be at upper end of aerobic
			scale_hr_param = 150			
			param2 = scale_hr_param * (1 / self.get_bucket_hr_at(token_power))
		return param2		


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
	def print_time_analysis(self, pdf_pages):
		"""Function writes results for given ride object to pdf page"""
	
		print "Printing time analysis data for ",self.fileName

		box_size = 185		# box_size < 90 often produce non-stationary variables
		endog_var = self.get_box_list(list(self.ride_dataFrame.Hrate.fillna(method='bfill')),box_size)
		# endog_var = self.get_box_list(list(self.ride_dataFrame.Hrate.fillna(method='bfill') - self.ride_dataFrame.Hrate.fillna(method='bfill').mean()),box_size)
		endog_var = self.detrend_list(endog_var)
		exog_var = self.get_box_list(list(self.ride_dataFrame.Watts.fillna(method='ffill')),box_size)

		diff_endog = self.difference_list(endog_var)

		p = 1
		q = 0
		d = 0
		model_order = (p,d,q)
		model = arma.ARIMA(endog_var,order=model_order,exog=exog_var)
		model_results = model.fit()
		prediction = model_results.fittedvalues
		residuals = model_results.resid
		min_resid = numpy.min(residuals)
		power_mean = numpy.mean(exog_var)
		resid_mean = numpy.mean(numpy.abs(residuals))

		dev_from_pw_mean = numpy.abs(exog_var - power_mean)
		resid_dev_from_mean = residuals - power_mean
		exog_dev_from_mean = exog_var - power_mean

		error_list = (residuals/resid_mean)  * (exog_var / power_mean)
		max_error = numpy.max(error_list)

		x_list = numpy.arange(0,len(exog_var))
		x_list1 = numpy.arange(0,len(prediction))
		x_list2 = numpy.arange(0,len(diff_endog))
		print model_results.maparams
		print model_results.arparams

		arx_title = "ARX model for " + self.fileName 
		canvas = plt.figure()
		ax_arma = canvas.add_subplot(311)
		ax_arma.set_title(arx_title)
		ax_arma.plot(x_list,endog_var,label='HR Data',color='red')
		ax_arma.plot(x_list1,prediction,label='Prediction',color='green')
		ax_arma.legend(loc=2, borderaxespad=0.,fontsize= 'xx-small')
		# Save and close
		# pdf_pages.savefig(canvas,orientation='portrait')	
		plt.close()	

		exert_title = "Scaled under-shooting for " + self.fileName 
		pw_title = "Power impulses for " + self.fileName 
		# resid_canvas = plt.figure()
		ax_resid = canvas.add_subplot(312)
		ax_resid.set_title(exert_title)
		ax_pw = canvas.add_subplot(313)
		ax_pw.set_title(pw_title)
		ax_resid.plot(x_list1,error_list,label='Residuals',color='green')
		ax_resid.set_ylim(0,1.1*max_error)
		ax_pw.plot(x_list1,exog_var,label='Power',color='red')
		ax_resid.legend(loc=2, borderaxespad=0.,fontsize= 'xx-small')
		ax_pw.legend(loc=2, borderaxespad=0.,fontsize= 'xx-small')

		canvas.tight_layout()

		pdf_pages.savefig(canvas,orientation='portrait')	

		plt.close()	

	#---------------------------------------------------
	def detrend_list(self, original_list):
		x = xrange(0, len(original_list))
		slope, intercept, r_value, p_value, std_err = st.linregress(x,original_list)
		trend_list = [(x * slope + intercept) for x in original_list] 
		detrended_list = [val1 - val2 for val1, val2 in zip(original_list, trend_list)]
		return detrended_list



	#---------------------------------------------------
	def difference_list(self, original_list):
		d1 = original_list[1:]
		d2 = original_list[:-1]
		diff_list = [val1 - val2 for val1, val2 in zip(d1, d2)]
		return diff_list



	#---------------------------------------------------
	def get_box_list(self, original_list, box_length):
		box_list = [numpy.mean(original_list[x:x+box_length]) for x in xrange(0, len(original_list), box_length)]
		return box_list

	#---------------------------------------------------
	def print_regressions(self, pdf_pages):
		"""Function writes results for given ride object to pdf page"""
	
		print "Printing regression data for ",self.fileName

		minutes_per_tick = self.ride_dataFrame.Minutes[1] - self.ride_dataFrame.Minutes[0]
		x_time_plots = numpy.arange(len(self.ride_dataFrame.Minutes)) * minutes_per_tick

		canvas = plt.figure()

		box_length = 90
		hr_boxes = [self.untrimmed_hr[x:x+box_length] for x in xrange(0, len(self.untrimmed_hr), box_length)]
		pw_boxes = [self.untrimmed_pw[x:x+box_length] for x in xrange(0, len(self.untrimmed_pw), box_length)]
		hr_means = [numpy.mean(self.ride_dataFrame.Hrate.fillna(method='bfill').fillna(method='ffill')[x:x+box_length]) for x in xrange(0, len(self.ride_dataFrame.Hrate), box_length)]
		pw_means = [numpy.mean(self.ride_dataFrame.Watts.fillna(method='bfill').fillna(method='ffill')[x:x+box_length]) for x in xrange(0, len(self.ride_dataFrame.Watts), box_length)]
		# Power-Time and Hrate-time curves (overlaid)

		pw_t_ax = canvas.add_subplot(211)
		pw_t_ax2 = pw_t_ax.twinx()
		pw_title = "Power-Hrate response for " + self.fileName 
		pw_t_ax.set_title(pw_title)
		pw_t_ax.set_ylabel('Hrate (bpm)')
		pw_t_ax.set_xlabel('Time (minutes)')
		x_list = numpy.arange(len(self.untrimmed_hr)) * minutes_per_tick
		box_pos = x_list[0::box_length]
		pw_t_plot2 = pw_t_ax2.boxplot(pw_boxes, sym='',positions=box_pos)
		pw_t_ax2.xaxis.cla()
		pw_t_ax2.set_ylabel('Power (W)')
		pw_t_ax2.yaxis.label.set_color('blue')
		pw_t_plot1 = pw_t_ax.plot(x_list, self.untrimmed_hr, color='red')
		pw_t_ax.yaxis.label.set_color('red')
		pw_t_ax.tick_params(axis='y', colors='red')
		pw_t_ax2.tick_params(axis='y', colors='blue')

		# Hrate-Power Gaussian kde
		x = pw_means
		y = hr_means		
		power_hr_slope, intercept, r_value, p_value, std_err = st.linregress(x,y)
		xmin = numpy.min(x)
		xmax = numpy.max(x)
		ymin = numpy.min(y)
		ymax = numpy.max(y)
		values = numpy.vstack([x,y])
		kernel = st.gaussian_kde(values)
		X, Y = numpy.mgrid[xmin:xmax:100j, ymin:ymax:100j]
		positions = numpy.vstack([X.ravel(), Y.ravel()])
		Z = numpy.reshape(kernel(positions).T, X.shape)
		distr_ax = canvas.add_subplot(212)
		distr_title = "Power-Hrate distr. with regression (r^2:" + "{:10.3f}".format(r_value*r_value) + ")"
		distr_ax.set_ylabel('Hrate (bpm)')
		distr_ax.set_xlabel('Power (Watts)')
		distr_ax.set_title(distr_title)
		distr_ax.imshow(numpy.rot90(Z), cmap=plt.cm.gist_earth_r,
								extent=[xmin, xmax, ymin, ymax])
		fitness_trend = [(x_it * power_hr_slope + intercept) for x_it in x] 
		distr_ax.axis('auto')
		distr_ax.plot(x,fitness_trend, linewidth=2, color='red')

		canvas.tight_layout()

		# Save and close
		pdf_pages.savefig(canvas,orientation='portrait')	
		plt.close()	




#==================================================================================================#
#==================================================================================================#
#==================================================================================================#
class analysis_driver:
	"""Class orchestrates analysis of available workout csv files"""
	#---------------------------------------------------
	def __init__(self):
		self.ride_obj_list = []
		self.working_directory = "./"
		self.print_file = False
		self.parse_sys_in()

	#---------------------------------------------------
	def parse_sys_in(self):
		"""Function parses sys in commands"""
		usage_string = "Usage: analyze_rides.py -d <working_directory> -p <decimal_precision> -a\n"
		usage_string += "\t-d <directory> specifies directory to look for workout files.\n"
		usage_string += "\t-p turns on pdf generation.\n"
		# Parse input arguments
		try:
			opts, args = getopt.getopt(sys.argv[1:],'d:hp')
		except getopt.GetoptError as err:
			print str(err)
			print usage_string
			sys.exit(2)
		for opt, arg in opts:
			if opt == "-h":
				print usage_string
				sys.exit()
			elif opt == '-p':
				self.print_file = True
			elif opt == '-d':
				self.working_directory =  str(arg)
			else:
				"Did not recognize option:", str(opt)
				print usage_string
	

	#---------------------------------------------------
	def analyze_files(self):
		"""Function analyzes all available workout csv files"""
		# Find workout files
		file_name_list = []
		match_string = self.working_directory + "/*.csv"
		numFiles = len(glob.glob(match_string))
		if not numFiles:
			print "No workout files were found in directory: " , str(os.getcwd())
			print "Specify a directory for workout values with the command -d <directory>"
			sys.exit()

		# Loop over all the files and analyze...
		# numFiles = 1
		for it in range(1,numFiles+1):
			lowest_accepted_r2 = 0
			# Determine filename
			fileName = self.working_directory + "/workouts"
			if it > 1:
				fileName += str(it)
			fileName += ".csv"
			print "Analyzing data in" , fileName, "..."

			ride_obj = rideData(fileName, it)
			ride_obj.clean_ride_data()
			self.ride_obj_list.append(ride_obj)

			if not ride_obj.has_good_data:
				print "     Power data corrupted in file ", fileName
				print "     File could not be analyzed."
				continue


	#---------------------------------------------------
	def show_results(self):
		"""Function shows results of analysis of workout files"""

		if self.print_file:
			print_filename = "workout_analysis.pdf"
			pdf_pages = PdfPages(print_filename)
			self.print_fitness_trend(pdf_pages)
			self.print_individual_workouts(pdf_pages)
			pdf_pages.close()	


	
	#---------------------------------------------------
	def print_individual_workouts(self,pdf_pages):
		"""Function prints a page summarizing each recorded workout"""
		for it, ride_obj in enumerate(self.ride_obj_list):
			if ride_obj.has_good_data:
				ride_obj.print_regressions(pdf_pages)
				ride_obj.print_time_analysis(pdf_pages)	


	#---------------------------------------------------
	def print_fitness_trend(self, pdf_pages):
		"""Function prints summary page for all recorded workouts"""
		fitness_list = []
		param1_list = []
		param2_list = []
		valid_files_list = []
		# Perform linear regression on fitness values to show progress (or lack thereof)
		for it, ride_obj in enumerate(self.ride_obj_list):
			if ride_obj.has_good_data:
				fitness_list.append(ride_obj.get_fitness_param())
				param1_list.append(ride_obj.get_param1())
				param2_list.append(ride_obj.get_param2())
				valid_files_list.append(it)

		# Run regression on fitness values to determine progress
		fitness_slope, intercept, r_value, p_value, std_err = st.linregress(valid_files_list,fitness_list)
		print "\nAnalysis report for fitness regression:"
		print '   std_err:',std_err, ', r^2:' , r_value*r_value , ', p:', p_value,', slope:',fitness_slope

		# Show fitness plot with regression
		canvas = plt.figure()
		canvas.suptitle('Summary')
		ax_p1p2 = canvas.add_subplot(221)
		ax_p1p2.set_title('Parameter overview')
		ax_p1p2.set_xlabel('File number')
		ax_p1p2.set_ylabel('Param 1, 2 values')
		ax_p1p2.set_xlim([0, 1.1*max(valid_files_list)])
		ax_p1p2.scatter(valid_files_list,param1_list, color='blue', label='Param1')
		ax_p1p2.scatter(valid_files_list,param2_list, color='red', label='Param2')
		ax_p1p2.legend(loc=2, borderaxespad=0.,fontsize= 'xx-small')

		# Plot fitness velocity on fitness scatter plot
		ax_fitScat = canvas.add_subplot(222)
		ax_fitScat.set_title('Fitness velocity:'+"{:10.5f}".format(fitness_slope))
		ax_fitScat.set_xlabel('File number')
		ax_fitScat.set_ylabel('Fitness values')
		ax_fitScat.set_xlim([0, 1.1*max(valid_files_list)])
		ax_fitScat.scatter(valid_files_list,fitness_list, label='Fitness values')
		fitness_trend = [(x * fitness_slope + intercept) for x in valid_files_list] 
		ax_fitScat.plot(valid_files_list,fitness_trend, label='Curve fit')
		ax_fitScat.legend(loc=2, borderaxespad=0.,fontsize= 'xx-small')

		# Plot fitness acceleration on fitness line plot
		ax_plot_score = canvas.add_subplot(212)
		ax_plot_score.set_xlabel('File number')
		ax_plot_score.set_ylabel('Fitness values')
		ax_plot_score.set_xlim([0, 1.1*max(valid_files_list)])
		ax_plot_score.plot(valid_files_list,fitness_list, label='Fitness values')
		coeff = numpy.polyfit(valid_files_list,fitness_list,2)
		fitness_curve = [(coeff[0]*x*x + coeff[1]*x + coeff[2]) for x in valid_files_list] 
		ax_plot_score.set_title('Fitness acceleration:'+"{:10.7f}".format(2*coeff[0]))
		ax_plot_score.plot(valid_files_list,fitness_curve, label='Curve fit')
		ax_plot_score.legend(loc=2, borderaxespad=0.,fontsize= 'xx-small')

		canvas.tight_layout()

		pdf_pages.savefig(canvas, orientation='portrait')	
		plt.close()


#==================================================================================================#
#==================================================================================================#
#==================================================================================================#
if __name__ == "__main__":
	# Instantiate driver object
	rides_driver = analysis_driver()
	# Run analysis
	rides_driver.analyze_files()
	print "\nCompleted analysis. Formatting results..."
	rides_driver.show_results()
	print "\nExiting program..."
