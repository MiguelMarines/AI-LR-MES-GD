# Libraries
import numpy as np													# Easier operations with arrays.

# ====================================================================================================================================== #
#                                                          HYPOTESIS FUNCTION                                                            #
# ====================================================================================================================================== #
# Hypothesis Function: y = θ₁x₁ + θ₂x₂ + θ₃x₃ ... + b

# theta -> Parameter θ or m.
# x_feature -> Data x inputs of features.
# b -> bias.
# y -> result or prediction.

def hypothesis_function(thetas, x_features_inputs):
	summation = 0                                                    # Acumulates the computations.
	for i in range(len(thetas)):                                     # Executes one case from the data set (1 Row).
		summation = summation + (thetas[i] * x_features_inputs[i])   # Computation of the hypothesis function.
	return summation                                                 # Returns the y (result value), obtained by the hypothesis function.

# ====================================================================================================================================== #





# ====================================================================================================================================== #
#                                                      MEAN SQUARE ERROR FUNCTION                                                        #
# ====================================================================================================================================== #
# Mean Square Error Function: MSE = 1/n * Σ(X₁ - Y₂)²

# y_results -> Expected Results
# r_results -> Real Results

def mean_square_error_function(y_results, r_results):
    summation = 0                                       			# Variable to store the summation of differences.
    for i in range(len(y_results)):                    				# Mean square error function computation with: MSE = 1/n * Σ(X₁ - Y₂)²
        difference = y_results[i] - r_results[i]
        squared_difference = difference**2
        summation = summation + squared_difference

    mean_square_error = summation / len(r_results)
	
    return mean_square_error                           				# Returns the mean square error value.

# ====================================================================================================================================== #





# ====================================================================================================================================== #
#                                                       GRADIENT DESCENT FUNCTION                                                        #
# ====================================================================================================================================== #
# Gradient Descent Function: θj = θj - α/m Σ[(hθ(Xi) - Y)Xi]

# theta -> Parameter θ or m.
# x_feature -> Data x inputs of features.
# b -> bias.
# y_results -> List containing the corresponding real result for each sample.
# alfa -> Learning rate.

def gradient_descent_function(thetas, x_features_inputs, y_results, alfa):
	gradient_descent = list(thetas)																# Creates list of the lenght of the thetas.
	for i in range(len(thetas)):
		summation = 0
		for j in range(len(x_features_inputs)):													# Gradient descendant computation.
			error = hypothesis_function(thetas, x_features_inputs[j]) - y_results[j]
			summation = summation + (error * x_features_inputs[j][i])
		gradient_descent[i] = thetas[i] - (alfa * (1/len(x_features_inputs)) * summation)

	return gradient_descent																		# Returns the errors from the thetas and the bias.

# ====================================================================================================================================== #





# ====================================================================================================================================== #
#                                                           Short Data Set                                                               #
# ====================================================================================================================================== #

# IMPORT DATA FROM THE DATASET
x_features_inputs = np.loadtxt('/Users/miguelangelmarinesolvera/Downloads/AI_Project/SAT_Scores_Dataset.data', dtype = int, delimiter=',')
x_features_inputs = np.delete(x_features_inputs, 3, axis=1)
x_features_inputs = x_features_inputs.astype('int')
#print(x_features_inputs)

y_results = np.loadtxt("/Users/miguelangelmarinesolvera/Downloads/AI_Project/SAT_Scores_Dataset.data", usecols = 3, dtype = int, delimiter=",")
#print(y_results)

alfa = 0.1  																# Learning rate.
thetas = [0.25,1.15,0]														# Initial thetas.


# HYPOTHESIS FUNCTION
print("\nHypothesis Function:")

y_hypothesis_function_results = list(y_results)

for i in range(len(x_features_inputs)):
	hypothesis = 0
	hypothesis = hypothesis_function(thetas, x_features_inputs[i])
	print("\nStudent (Case)", i + 1, ": ",hypothesis)
	y_hypothesis_function_results[i] = hypothesis





# MEAN SQUARE ERROR
print("\n\n\nY Computed and Y Real Results:")
print("\nY - Hypothesis Function Results: ", *y_hypothesis_function_results)
print("\nY - Real Results: ", *y_results)


mean_square_error = mean_square_error_function(y_hypothesis_function_results, y_results)
print("\n\nMean Square Error: ", mean_square_error)





# GRADIENT DESCENT
print("\n\n\nGradient Descent Function: ")
old_thetas = list(thetas)														# Creates list of the lenght of the thetas.
print("\nOld Thetas: ", thetas)																	# Prints the current thetas.

thetas = gradient_descent_function(thetas, x_features_inputs, y_results, alfa)	# Executes the gradient descendent function, which executes the hypothesis function, in order to find the parameters (thetas).
print("\nNew Thetas", thetas)

print()
print()

# ====================================================================================================================================== #





# # ====================================================================================================================================== #
# #                                                               Full Data Set                                                            #
# # ====================================================================================================================================== #


# # IMPORT DATA FROM THE DATASET
# x_features_inputs = np.loadtxt('/Users/miguelangelmarinesolvera/Downloads/AI_Project/SAT_Scores_Dataset1.data', dtype = int, delimiter=',')
# x_features_inputs = np.delete(x_features_inputs, 3, axis=1)
# x_features_inputs = x_features_inputs.astype('int')
# #print(x_features_inputs)

# y_results = np.loadtxt("/Users/miguelangelmarinesolvera/Downloads/AI_Project/SAT_Scores_Dataset1.data", usecols = 3, dtype = int, delimiter=",")
# #print(y_results)

# alfa = 0.1  																# Learning rate.
# thetas = [0.25,1.15,0]														# Initial thetas.


# # HYPOTHESIS FUNCTION
# print("\nHypothesis Function:")

# y_hypothesis_function_results = list(y_results)

# for i in range(len(x_features_inputs)):
# 	hypothesis = 0
# 	hypothesis = hypothesis_function(thetas, x_features_inputs[i])
# 	print("\nStudent (Case)", i + 1, ": ",hypothesis)
# 	y_hypothesis_function_results[i] = hypothesis





# # MEAN SQUARE ERROR
# print("\n\n\nY Computed and Y Real Results:")
# print("\nY - Hypothesis Function Results: ", *y_hypothesis_function_results)
# print("\nY - Real Results: ", *y_results)


# mean_square_error = mean_square_error_function(y_hypothesis_function_results, y_results)
# print("\n\nMean Square Error: ", mean_square_error)





# # GRADIENT DESCENT
# print("\n\n\nGradient Descent Function: ")
# old_thetas = list(thetas)														# Creates list of the lenght of the thetas.
# print("\nOld Thetas: ", thetas)																	# Prints the current thetas.

# thetas = gradient_descent_function(thetas, x_features_inputs, y_results, alfa)	# Executes the gradient descendent function, which executes the hypothesis function, in order to find the parameters (thetas).
# print("\nNew Thetas", thetas)

# print()
# print()

# # ====================================================================================================================================== #

