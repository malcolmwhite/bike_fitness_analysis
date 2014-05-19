import scipy.stats as st
import matplotlib.pyplot as plt
import sys, getopt, os, math, glob
import numpy as np
import pandas.io.parsers as parse
from pandas import DataFrame as df
import statsmodels.nonparametric.kernel_regression as kr
import statsmodels.tsa.arima_model as arma
import statsmodels.tsa.tsatools as tsa
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.tight_layout as plt_l


#==================================================================================================#
#==================================================================================================#
#==================================================================================================#
class printRideData:
	"""Class reads workout .csv files and processes / analyzes data"""
	#---------------------------------------------------
	def __init__(self, ride_obj_list, print_filename):
		# Initialize variables
		self.ride_obj_list = ride_obj_list
		self.print_filename = print_filename
		self.endog_var = []
		self.exog_var = []
		self.mle_prediction = []
		self.mle_param = None
		self.mle_resid = []
		self.mle_pos_resid = []

		
	#---------------------------------------------------
	#---------------------------------------------------
	#---------------------------------------------------
	def __enter__(self):
		self.pdf_pages = PdfPages(self.print_filename)
		return self


	#---------------------------------------------------
	#---------------------------------------------------
	#---------------------------------------------------
	def __exit__(self, type, value, traceback):
		self.pdf_pages.close()	

	#---------------------------------------------------
	#---------------------------------------------------
	#---------------------------------------------------
	def print_individual_workouts(self):
		"""Function prints a page summarizing each recorded workout"""
		for it, ride_obj in enumerate(self.ride_obj_list):
			if ride_obj.has_good_data:
				self.print_regressions(ride_obj)
				self.print_time_analysis(ride_obj)	


	#---------------------------------------------------
	#---------------------------------------------------
	#---------------------------------------------------
	def print_fitness_trend(self):
		"""Function prints summary page for all recorded workouts"""
		fitness_list = []
		param1_list = []
		param2_list = []
		param3_list = []
		param4_list = []
		resid_list = []
		valid_files_list = []

		title_plt = plt.figure()
		title_plt.text(0.05,0.7,"Fitness evaluation using (1) linear regression of time avgeraged HR v. Power and")
		title_plt.text(0.31,0.65,"(2) MLE fit of HRate - Power relationship in time")
		self.pdf_pages.savefig(title_plt, orientation='portrait')	


		# Perform linear regression on fitness values to show progress (or lack thereof)
		for it, ride_obj in enumerate(self.ride_obj_list):
			if ride_obj.has_good_data:
				fitness_list.append(ride_obj.get_fitness_param())
				param1_list.append(ride_obj.get_param1())
				param2_list.append(ride_obj.get_param2())
				param3_list.append(ride_obj.get_param3())
				param4_list.append(ride_obj.get_param4())
				resid_list.append(ride_obj.get_sum_pos_resid())
				valid_files_list.append(it)

		# Run regression on fitness values to determine progress
		fitness_slope, intercept, r_value, p_value, std_err = st.linregress(valid_files_list,fitness_list)
		print "\nAnalysis report for fitness regression:"
		print '   std_err:',std_err, ', r^2:' , r_value*r_value , ', p:', p_value,', slope:',fitness_slope

		# Show fitness plot with regression
		canvas = plt.figure()
		ax_p1p2 = canvas.add_subplot(221)
		ax_p1p2.set_title('Parameter overview')
		ax_p1p2.set_xlabel('File number')
		ax_p1p2.set_ylabel('Score')
		ax_p1p2.set_xlim([0, 1.1*max(valid_files_list)])
		ax_p1p2.scatter(valid_files_list,param1_list, color='blue', label='Power Density')
		ax_p1p2.scatter(valid_files_list,param2_list, color='red', label='Regression')
		ax_p1p2.scatter(valid_files_list,param3_list, color='green', label='MLE')
		ax_p1p2.scatter(valid_files_list,param4_list, color='orange', label='Endurance')
		ax_p1p2.legend(loc=2, borderaxespad=0.,fontsize= 'xx-small')

		# Plot fitness velocity on fitness scatter plot
		ax_fitScat = canvas.add_subplot(222)
		ax_fitScat.set_title('Fitness velocity:'+"{:10.5f}".format(fitness_slope))
		ax_fitScat.set_xlabel('File number')
		ax_fitScat.set_ylabel('Score')
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
		coeff = np.polyfit(valid_files_list,fitness_list,2)
		fitness_curve = [(coeff[0]*x*x + coeff[1]*x + coeff[2]) for x in valid_files_list] 
		ax_plot_score.set_title('Fitness acceleration:'+"{:10.7f}".format(2*coeff[0]))
		ax_plot_score.plot(valid_files_list,fitness_curve, label='Curve fit')
		ax_plot_score.legend(loc=2, borderaxespad=0.,fontsize= 'xx-small')

		canvas.tight_layout()

		self.pdf_pages.savefig(canvas, orientation='portrait')	
		plt.close()



	#---------------------------------------------------
	#---------------------------------------------------
	#---------------------------------------------------
	def print_time_analysis(self, ride_obj):
		"""Function writes results for given ride object to pdf page"""
	
		print "Printing time analysis data for ",ride_obj.fileName

		endog_var = ride_obj.endog_var
		exog_var = ride_obj.exog_var

		prediction = ride_obj.mle_prediction
		residuals = ride_obj.mle_resid

		error_list = ride_obj.mle_pos_resid
		max_error = np.max(error_list)
		error_sum = np.sum(error_list)
		error_density = error_sum / len(error_list)

		endog_x_list = np.arange(0,len(endog_var))
		exog_x_list = np.arange(0,len(exog_var))
		prediction_x_list = np.arange(0,len(prediction))
		error_x_list = np.arange(0,len(error_list))

		canvas = plt.figure()

		arx_title = "MLE fit for " + ride_obj.fileName 
		ax_arma = canvas.add_subplot(311)
		ax_arma.set_title(arx_title)
		ax_arma.plot(endog_x_list, endog_var,label='HR Data',color='red')
		ax_arma.plot(prediction_x_list, prediction,label='Prediction',color='green')
		ax_arma.legend(loc=2, borderaxespad=0.,fontsize= 'xx-small')
		ax_arma.set_xlabel('Time (min)')
		ax_arma.set_ylabel('Detrended Hrate (bpm)')
		tick_locs, tick_labels =  plt.xticks()
		tick_labels = tick_locs*3
		plt.xticks(tick_locs, tick_labels)
		ax_arma.set_xlim(0,len(prediction_x_list))

		exert_title = "Positive error from MLE prediction" 
		ax_resid = canvas.add_subplot(312)
		ax_resid.set_title(exert_title)
		ax_resid.plot(error_x_list, error_list,label='Residuals',color='blue')
		ax_resid.set_xlabel('Time (min)')
		ax_resid.set_ylabel('Residual (bpm)')
		plt.xticks(tick_locs, tick_labels)
		ax_resid.set_xlim(0,len(error_x_list))
		ax_resid.set_ylim(0,1.1*max_error)
		ax_resid.legend(loc=2, borderaxespad=0.,fontsize= 'xx-small')

		# ax_pw = ax_resid.twinx()
		# # pw_title = "Power used as exog. var. for MLE" 
		# # ax_pw.set_title(pw_title)
		# plt.xticks(tick_locs, tick_labels)
		# ax_pw.set_ylabel('Power (W)')
		# ax_pw.plot(exog_x_list,exog_var,label='Power',color='red')
		# ax_pw.legend(loc=2, borderaxespad=0.,fontsize= 'xx-small')
		# ax_pw.set_xlim(0,len(exog_x_list))

		# Fitness parameters
		ax_param = canvas.add_subplot(313)
		param_title = "Overview of fitness parameters"
		ax_param.set_title(param_title)
		param1 = ride_obj.get_param1()
		param2 = ride_obj.get_param2()
		param3 = ride_obj.get_param3()
		param4 = ride_obj.get_param4()
		fitness = ride_obj.get_fitness_param()
		ax_param.text(0.1,0.7,"Parameter 1: "+str(param1)[0:6])		
		ax_param.text(0.1,0.5,"Parameter 2: "+str(param2)[0:6])		
		ax_param.text(0.1,0.3,"Parameter 3: "+str(param3)[0:6])		
		ax_param.text(0.1,0.1,"Parameter 4: "+str(param4)[0:6])		
		ax_param.text(0.6,0.1,"Summary Parameter: "+str(fitness)[0:6])	
		ax_param.text(0.6,0.5,"MLE error sum: "+str(error_sum)[0:6])	
		ax_param.text(0.6,0.3,"MLE error density: "+str(error_density)[0:6])	
		ax_param.set_xticks([])	
		ax_param.set_yticks([])	

		canvas.tight_layout()

		self.pdf_pages.savefig(canvas,orientation='portrait')	

		plt.close()	


	#---------------------------------------------------
	#---------------------------------------------------
	#---------------------------------------------------
	def print_regressions(self, ride_obj):
		"""Function writes results for given ride object to pdf page"""
	
		print "Printing regression data for ",ride_obj.fileName

		minutes_per_tick = ride_obj.ride_dataFrame.Minutes[1] - ride_obj.ride_dataFrame.Minutes[0]
		x_time_plots = np.arange(len(ride_obj.ride_dataFrame.Minutes)) * minutes_per_tick

		# Initialize plotting figure
		canvas = plt.figure()

		# Get averaged variables
		box_length = 60
		hr_boxes = [ride_obj.untrimmed_hr[x:x+box_length] for x in xrange(0, len(ride_obj.untrimmed_hr), box_length)]
		pw_boxes = [ride_obj.untrimmed_pw[x:x+box_length] for x in xrange(0, len(ride_obj.untrimmed_pw), box_length)]
		hr_means = [np.mean(ride_obj.ride_dataFrame.Hrate.shift(-23).fillna(method='bfill').fillna(method='ffill')[x:x+box_length]) for x in xrange(0, len(ride_obj.ride_dataFrame.Hrate), box_length)]
		pw_means = [np.mean(ride_obj.ride_dataFrame.Watts.fillna(method='bfill').fillna(method='ffill')[x:x+box_length]) for x in xrange(0, len(ride_obj.ride_dataFrame.Watts), box_length)]
		
		# Power-Time and Hrate-time curves (overlaid)
		pw_t_ax = canvas.add_subplot(211)
		pw_t_ax2 = pw_t_ax.twinx()
		pw_title = "Power-Hrate response for " + ride_obj.fileName 
		pw_t_ax.set_title(pw_title)
		pw_t_ax.set_ylabel('Hrate (bpm)')
		pw_t_ax.set_xlabel('Time (minutes)')
		x_list = np.arange(len(ride_obj.untrimmed_hr)) * minutes_per_tick
		box_pos = x_list[0::box_length]
		pw_t_plot2 = pw_t_ax2.boxplot(pw_boxes, sym='',positions=box_pos)
		pw_t_ax2.xaxis.cla()
		pw_t_ax2.set_ylabel('Power (W)')
		pw_t_ax2.yaxis.label.set_color('blue')
		pw_t_plot1 = pw_t_ax.plot(x_list, ride_obj.untrimmed_hr, color='red')
		pw_t_ax.yaxis.label.set_color('red')
		pw_t_ax.tick_params(axis='y', colors='red')
		pw_t_ax2.tick_params(axis='y', colors='blue')

		# Hrate-Power Gaussian kde
		x = pw_means
		y = hr_means		
		power_hr_slope, intercept, r_value, p_value, std_err = st.linregress(x,y)
		xmin = np.min(x)
		xmax = np.max(x)
		ymin = np.min(y)
		ymax = np.max(y)
		values = np.vstack([x,y])
		kernel = st.gaussian_kde(values)
		X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
		positions = np.vstack([X.ravel(), Y.ravel()])
		Z = np.reshape(kernel(positions).T, X.shape)
		distr_ax = canvas.add_subplot(212)
		distr_title = "Power-Hrate distr. with regression (r^2:" + "{:10.3f}".format(r_value*r_value) + ")"
		distr_ax.set_ylabel('Hrate (bpm)')
		distr_ax.set_xlabel('Power (Watts)')
		distr_ax.set_title(distr_title)
		distr_ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r,
								extent=[xmin, xmax, ymin, ymax])
		fitness_trend = [(x_it * power_hr_slope + intercept) for x_it in x] 
		distr_ax.axis('auto')
		distr_ax.plot(x,fitness_trend, linewidth=2, color='red')

		canvas.tight_layout()

		# Save and close
		self.pdf_pages.savefig(canvas,orientation='portrait')	
		plt.close()	


	#---------------------------------------------------
	#---------------------------------------------------
	#---------------------------------------------------
	def print_breakpt_regressions(self, ride_obj):
		"""Function writes results for given ride object to pdf page"""
	
		print "Printing regression data for ",ride_obj.fileName

		minutes_per_tick = ride_obj.ride_dataFrame.Minutes[1] - ride_obj.ride_dataFrame.Minutes[0]
		x_time_plots = np.arange(len(self.ride_dataFrame.Minutes)) * minutes_per_tick

		# Initialize plotting figure
		canvas = plt.figure()

		shifted_hr = ride_obj.ride_dataFrame.Hrate.shift(-23).fillna(method='bfill').fillna(method='ffill')
		full_pw_list = ride_obj.ride_dataFrame.Watts.fillna(method='bfill').fillna(method='ffill')

		# Get averaged variables
		box_length = 90
		pw_breakpt = 120
		hr_boxes = [ride_obj.untrimmed_hr[x:x+box_length] for x in xrange(0, len(ride_obj.untrimmed_hr), box_length)]
		pw_boxes = [ride_obj.untrimmed_pw[x:x+box_length] for x in xrange(0, len(ride_obj.untrimmed_pw), box_length)]
		
		hr_means = [np.mean(shifted_hr[x:x+box_length]) for x in xrange(0, len(ride_obj.ride_dataFrame.Hrate), box_length)]
		pw_means = [np.mean(full_pw_list[x:x+box_length]) for x in xrange(0, len(ride_obj.ride_dataFrame.Watts), box_length)]
		
		hr_means_low = []
		hr_means_high = []
		pw_means_low =  []
		pw_means_high = []

		for index, avg_pw in enumerate(pw_means):
			if avg_pw < pw_breakpt:
				hr_means_low.append(hr_means[index])
				pw_means_low.append(avg_pw)
			else:
				hr_means_high.append(hr_means[index])
				pw_means_high.append(avg_pw)
		
		# Power-Time and Hrate-time curves (overlaid)
		pw_t_ax = canvas.add_subplot(211)
		pw_t_ax2 = pw_t_ax.twinx()
		pw_title = "Power-Hrate response for " + ride_obj.fileName 
		pw_t_ax.set_title(pw_title)
		pw_t_ax.set_ylabel('Hrate (bpm)')
		pw_t_ax.set_xlabel('Time (minutes)')
		x_list = np.arange(len(ride_obj.untrimmed_hr)) * minutes_per_tick
		box_pos = x_list[0::box_length]
		pw_t_plot2 = pw_t_ax2.boxplot(pw_boxes, sym='',positions=box_pos)
		pw_t_ax2.xaxis.cla()
		pw_t_ax2.set_ylabel('Power (W)')
		pw_t_ax2.yaxis.label.set_color('blue')
		pw_t_plot1 = pw_t_ax.plot(x_list, ride_obj.untrimmed_hr, color='red')
		pw_t_ax.yaxis.label.set_color('red')
		pw_t_ax.tick_params(axis='y', colors='red')
		pw_t_ax2.tick_params(axis='y', colors='blue')

		# Hrate-Power Gaussian kde
		x = pw_means
		y = hr_means
		x_low = pw_means_low
		y_low = hr_means_low
		# print len(x_low), len(y_low)
		# print x_low[0][3]
		x_high = pw_means_high
		y_high = hr_means_high		
		power_hr_slope_low, intercept_low, r_value_low, p_value_low, std_err_low = st.linregress(x_low,y_low)
		power_hr_slope_high, intercept_high, r_value_high, p_value_high, std_err_high = st.linregress(x_high,y_high)
		# print x_low
		pw_x_intersection = (intercept_high - intercept_low) / (power_hr_slope_low - power_hr_slope_high) 
		# pw_x_intersection = pw_breakpt
		xmin = np.min(x)
		xmax = np.max(x)
		ymin = np.min(y)
		ymax = np.max(y)
		values = np.vstack([x,y])
		kernel = st.gaussian_kde(values)
		X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
		positions = np.vstack([X.ravel(), Y.ravel()])
		Z = np.reshape(kernel(positions).T, X.shape)
		distr_ax = canvas.add_subplot(212)
		distr_title = "Power-Hrate distr. with regression (r^2:" + "{:10.3f}".format(r_value_low*r_value_low) + ")"
		distr_ax.set_ylabel('Hrate (bpm)')
		distr_ax.set_xlabel('Power (Watts)')
		distr_ax.set_title(distr_title)
		distr_ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r,
								extent=[xmin, xmax, ymin, ymax])
		# x_low = xrange(0,int(pw_x_intersection)-1)
		# x_high = xrange(int(pw_x_intersection),int(pw_x_intersection)+len)
		print pw_x_intersection
		pw_x_intersection = int(pw_x_intersection)
		x_list_low = xrange(0,pw_x_intersection-1)
		x_list_high = xrange(pw_x_intersection,int(xmax))
		fitness_trend_low = [(x_it * power_hr_slope_low + intercept_low) for x_it in x_list_low] 
		fitness_trend_high = [(x_it * power_hr_slope_high + intercept_high) for x_it in x_list_high] 
		distr_ax.axis('auto')
		distr_ax.scatter(x_list_low,fitness_trend_low, color='red')
		distr_ax.scatter(x_list_high,fitness_trend_high, color='blue')

		canvas.tight_layout()

		# Save and close
		pdf_pages.savefig(canvas,orientation='portrait')	
		plt.close()	
