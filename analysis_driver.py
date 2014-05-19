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
import analyzeRideData as ard

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
	#---------------------------------------------------
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
			# Determine filename
			fileName = self.working_directory + "/workouts"
			if it > 1:
				fileName += str(it)
			fileName += ".csv"
			print "Analyzing data in" , fileName, "..."

			ride_obj = ard.analyzeRideData(fileName, it)
			ride_obj.clean_ride_data()
			ride_obj.perform_time_analysis()
			self.ride_obj_list.append(ride_obj)

			if not ride_obj.has_good_data:
				print "     Power data corrupted in file ", fileName
				print "     File could not be analyzed."
				continue


	#---------------------------------------------------
	#---------------------------------------------------
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
	#---------------------------------------------------
	#---------------------------------------------------
	def print_individual_workouts(self,pdf_pages):
		"""Function prints a page summarizing each recorded workout"""
		for it, ride_obj in enumerate(self.ride_obj_list):
			if ride_obj.has_good_data:
				ride_obj.print_regressions(pdf_pages)
				ride_obj.print_time_analysis(pdf_pages)	


	#---------------------------------------------------
	#---------------------------------------------------
	#---------------------------------------------------
	def print_fitness_trend(self, pdf_pages):
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
		pdf_pages.savefig(title_plt, orientation='portrait')	


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
