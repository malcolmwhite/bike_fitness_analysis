import sys, getopt, os, glob
import analyzeRideData as ard
import printRideData as prd

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
			with prd.printRideData(self.ride_obj_list, print_filename) as printer:
				printer.print_fitness_trend()
				printer.print_individual_workouts()
			


	


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
