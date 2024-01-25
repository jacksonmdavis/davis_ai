import pandas as pd
import numpy as np
from math import ceil, sqrt
from itertools import product
import matplotlib.pyplot as plt

from . import classifiers as davcls

# ## primary_stats_table
#
# primary_stats_table() accepts a pandas DataFrame with features to be summarized, as well as some optional parameters, and returns a pandas dataframe representing a table with various test statistics displayed.
# The DataFrame passed in should only contain data that should be included in the calculations (all values should be numerical).
#
# **Parameters**
#
# * **data** pd.DataFrame: the DataFrame containing feature data to be summarized.
# * **name** string, default None: a name to apply to the dataframe.
# * **trim** integer, default 1: number of values to trim *from each end* of the data when calculating Trimmed Mean
# * **atrim** float, default 0.05: percentage number of *total* values to trim from the data when calculating Alpha Trimmed Mean (rounded up to nearest even int, half removed from each side)
#
# **returns** pd.DataFrame: A dataframe with a suite of test statistics calculated for each column in the input dataframe.
def primary_stats_table(data: pd.DataFrame, name: str = None, trim: int = 1, atrim: float = 0.05):

    test_statistics = ['Min', 'Max', 'Mean', 'Trim_Mean', 'a_Trim_Mean', 'SD', 'Skewness', 'Kurtosis']

    stat_table = pd.DataFrame(dtype=float, index=(test_statistics))
    if (name != None):
        stat_table.columns.name = name
    
    for column in data:
        stat_table[column] = np.nan
    
    for column in stat_table:
        
        #calculate easy stats
        stat_table[column].loc['Min'] = data[column].min()
        stat_table[column].loc['Max'] = data[column].max()
        stat_table[column].loc['Mean'] = data[column].mean()
        
        #calculate trimmed mean
        temp_col = data[column].copy()
        for i in range(trim):
            temp_col = temp_col.drop(temp_col.idxmin())
            temp_col = temp_col.drop(temp_col.idxmax())
        stat_table[column].loc['Trim_Mean'] = temp_col.mean()

        #calculate alpha-trimmed mean
        atrim_val = ceil(len(data[column]) * atrim)
        atrim_val = (atrim_val + (atrim_val % 2)) // 2
        temp_col = data[column]
        for i in range(atrim_val):
            temp_col = temp_col.drop(temp_col.idxmin())
            temp_col = temp_col.drop(temp_col.idxmax())
        stat_table[column].loc['a_Trim_Mean'] = temp_col.mean()

        #calculate remaining easy stats
        stat_table[column].loc['SD'] = data[column].std()
        stat_table[column].loc['Skewness'] = data[column].skew()
        stat_table[column].loc['Kurtosis'] = data[column].kurtosis()

    return stat_table   



# ## key_merge_sort
#
# key_merge_sort() takes a list object with indexable values and returns a list with the same values sorted in ascending order based on the first index.
#
# **Parameters**
#
# * **data** list: list of tuple values to be sorted based on the first index of each value.  String indexes will be sorted lexographically.
#
# **returns** list: a list of the values in the input data sorted in ascending order based on the first index of each value.
def key_merge_sort(data: list, key: int = 0) -> list:
    
    length = len(data)
    if (length > 1):
        mid = length // 2

        data_L = key_merge_sort(data[:mid], key)
        data_R = key_merge_sort(data[mid:], key)

        output_list = []
        while (len(data_L) > 0) and (len(data_R) > 0):
            if data_L[0][key] <= data_R[0][key]:
                output_list.append(data_L.pop(0))
            else: 
                output_list.append(data_R.pop(0))
        
        while (len(data_L) > 0):
            output_list.append(data_L.pop(0))

        while (len(data_R) > 0):
            output_list.append(data_R.pop(0))

        return output_list

    else:  
        return data

assert(key_merge_sort([(1,'red'), (3,'blue'), (2,'red')]) == [(1,'red'), (2,'red'), (3,'blue')])
assert(key_merge_sort([(1,'red',3), (3,'blue',12), (2,'red',18)], key=0) == [(1,'red',3), (2,'red',18), (3,'blue',12)])
assert(key_merge_sort([(1,'red',3), (3,'blue',12), (2,'red',18)], key=2) == [(1,'red',3), (3,'blue',12), (2,'red',18)])


# ### get_covariance_m ###
#
# get_covariance_m() is a function that accepts a pandas dataframe and returns a covariance matrix for all features (columns) in the frame.
# The dataframe passed in must only include columns that should be included in the covariance matrix (all values must be numerical).
#
# Used by: m_distance()
#
# Parameters:
#
# df - dataframe: the pandas dataframe columns to be included in the covariance matrix.
#
# *returns* np.ndarray: the covariance matrix as a two-dimensional numpy array
def get_covariance_m(df: pd.DataFrame) -> np.ndarray:

    meanvector = df.mean()                                                                                                          

    #Creates empty matrix first; resizing matrix each loop is expensive.
    matrix = np.zeros((len(df.columns),len(df.columns)))                                                                            

    #Populates covariance matrix
    sample_size = df.shape[0]                                                                                                       
    for i, column1 in enumerate(df.columns):                                                                                        
        for j, column2 in enumerate(df.columns):
            matrix[i][j] = ((df[column1] - meanvector[column1]) * (df[column2] - meanvector[column2])).sum() / (sample_size - 1)    

    return matrix


# ## m_distance ##
#
# m_distance() accepts a data point array, a covariance matrix and a mean vector array, and returns that data point's Mahalanobis 
# Distance from the mean as a float.
# The vectors and matrix passed in should only contain data that should be included in the calculation (all values should be numerical).
#
# *Used by*: get_m_distances
#
# **Parameters**
#
# * **data** pd.Series: the data point for which Mahalanobis Distance will be calculated.
# * **cov_m** np.ndarray: the covariance matrix for the data that includes all features for the given data point (and no others).
# * **mean_vec** pd.Series: the vector containing the mean of each feature in the given data point.
#
# **returns** float: The Mahalanobis Distance for the given data point.
def m_distance(data, cov_m: np.ndarray, mean_vec) -> float:                                                       
                                                                                                                                        
    #sqrt((x − μ)T * Σ^−1 * (x − μ))
    distance = sqrt((data - mean_vec) @ np.linalg.inv(cov_m) @ (data - mean_vec))                                                       
    return distance


# ## get_m_distances ##
#
# get_m_distances() accepts a pandas DataFrame and returns a pandas Series containing the Mahalanobis Distance for each row entry in the dataframe.
# The dataframe passed in must only include columns that should be included in the calculation (all values should be numerical).
#
# **Uses**: get_covariance_m, m_distance
#
# **Parameters**
#
# * **df** dataframe: the pandas dataframe with rows for which the Mahalanobis Distance will be calculated.
#
# **returns** pd.Series: The Mahalanobis Distance for each row in the input.  Retains row indexing.
def get_m_distances(df: pd.DataFrame) -> pd.Series:                                     
                                                                                        
    meanvector = df.mean().to_numpy()                                                   
    cov_m = get_covariance_m(df)                                                        

    return df.apply(m_distance, axis = 1, raw = True, args= (cov_m, meanvector))        


# ## del_furthest_m_dist
#
# del_furthest_m_dist() accepts a pandas DataFrame and returns the DataFrame with the data point(s) with the largest Mahalanobis Distance (MD) from the mean removed.  MD will be recalculated for each successive removal.  Inputs should all be numerical, and all data points will be included in calculations, so separation by class should happen before calling if applicable.
#
# **Uses**: get_m_distances
#
# **Parameters**
#
# * **df** dataframe: the pandas dataframe containing data points to be removed.
# * **count** int, default 0: number of datapoints to be removed.
#
# **returns** dataframe: input dataframe with the data point(s) with the largest Mahalanobis Distance (MD) from the mean removed.
def del_furthest_m_dist(data: pd.DataFrame, count: int = 1) -> pd.DataFrame:
    
    for i in range(count):
        data = data.drop(get_m_distances(data).idxmax())

    return data

# Inputs list of indices, outputs list of k dictionaries, with each dictionary divided into 'train' and 'test' indices.
# If balance == True, any remainder after data is divided by k is ignored, so all tests will be of equal size.
def kfold_lists(indices: list, k: int = 5, balance: bool = False) -> dict:
    output = []
    indices = list(indices)
    length, remainder = divmod(len(indices),k)
    indices = list(np.random.permutation(indices))
    
    if balance == True and remainder != 0:
        del indices[-remainder:]

    test_start = 0
    test_end = 0
    for _ in range(k):
        outputdict = {}
        increment = int((balance == False) and (remainder >= 1))
        
        test_end = test_end + length + increment

        outputdict['test'] = indices[test_start:test_end]
        outputdict['train'] = indices[:test_start] + indices[test_end:]

        test_start = test_start + length + increment
        remainder = remainder - 1

        output.append(outputdict)
    
    return output

# Compares an array of predicted classes to an array of observed classes and returns an accuracy value.
# Confusion matrix capability to be added soon.
def classification_score(predicted: np.ndarray, observed: np.ndarray, verbose: bool = False)-> float:
    
    # This part could probably break in a number of horrible ways
    if type(observed) != np.ndarray:
        observed = np.array(observed)
    if type(predicted) != np.ndarray:
        predicted = np.array(predicted)

    if len(predicted) != len(observed):
        raise ValueError("Predicted and observed value lists must be the same length")
    
    classes = np.unique(observed)

    correct = 0
    count = 0

    for i in range(len(predicted)):

        if predicted[i] == observed[i]:
            correct += 1
        count += 1

    if verbose == False:
        return correct / count
    # confusion matrix to be implemented
    else:
        raise ValueError("Confusion matrix not yet implemented")


################################
#       1vAll Pipeline         #
################################

# Converts data to 1vsAll format for each feature for use with various classifiers
# Returns a series of predictions for each test observation, using the chosen classifier
def one_v_all_classify_pipeline(train_X: np.ndarray, train_y: np.ndarray, test_X: np.ndarray, *, 
                                classifier: str = 'SVM', kernel_type: str = 'rbf', arg: float = 0.15, C: float = 3.0) -> np.ndarray:


    results = {}
    classes = np.unique(train_y)

    if classifier == 'PARZEN':
        return davcls.parzen_kernel_classifier(train_X, train_y, test_X, kernel=kernel_type, h=arg)

    # Creates results dictionary: {'class1': [y], 'class2': [y], ...}
    for cls in classes:
        train_Y1 = train_y.copy()
        train_Y1[train_y != cls] = -1
        train_Y1[train_y == cls] = 1

        # Creates series of test results for each class based on the chosen classifier
        if classifier == 'SVM':
            model = davcls.SVM_train(train_X, train_Y1, kernel_type=kernel_type, arg=arg, C=C)
            test_result = davcls.SVM_1v1_classify(test_X, model)['y']

        elif classifier == 'RBFNN':
            model = davcls.rbf_NN_train(train_X, train_Y1, spread=arg)
            test_result = davcls.rbf_NN_1v1_classify_NoBias(test_X, model)['y']
        
        elif classifier == 'BAYES':
            model = davcls.gaussian_mix_model(train_X, train_Y1)
            test_result = davcls.bayes_classifier(model, test_X)

        else:
            raise ValueError('Invalid method')
        
        results[cls] = test_result     

    # Creates dataframe of results and finds the class with the highest probability for each test observation
    results_df = pd.DataFrame(results, dtype=float)
    predictions = results_df.idxmax(axis=1).to_numpy()
    
    return predictions


# This function takes in a dataframe of features, a series of classes, and the number of folds to use for k-fold cross validation
# I really need to build a whole pipeline using **kwargs
def K_fold_test(X, y, k, *, classifier: str = 'SVM', kernel_type: str = 'rbf', arg: float = 0.15, C: float = 3.0, verbose: bool = True, repeat: int = 1) -> np.ndarray:

    overall_accuracy = 0
    for i in range(repeat):
        k_dict = kfold_lists(indices=range(len(X)), k=k)

        avg_accuracy = 0

        for i, test in enumerate(k_dict):
            predicted = one_v_all_classify_pipeline(X[test['train'], :],
                                                    y[test['train']],
                                                    X[test['test'], :], 
                                                    classifier = classifier, 
                                                    C=C, 
                                                    kernel_type=kernel_type, 
                                                    arg=arg)
            accuracy = classification_score(predicted=predicted, observed=y[test['test']])
            
            avg_accuracy += accuracy
            if verbose:
                if classifier == 'BAYES':
                    print(f"Accuracy for test #{i+1}, {classifier} classifier: ", format((accuracy*100), ".2f"), "%", sep='')
                else:
                    print(f"Accuracy for test #{i+1}, {classifier} classifier with {kernel_type} kernel, C = {C}, arg(sigma/D)={arg}: ", 
                        format((accuracy*100), ".2f"), "%", sep='')
            
        avg_accuracy /= k
        print(f"Average accuracy for {k}-fold cross validation, {classifier} classifier with {kernel_type} kernel, C = {C}, arg(sigma/D)={arg}: {format((avg_accuracy*100), '.2f')}%")
        overall_accuracy += avg_accuracy

    if repeat > 1:
        overall_accuracy /= repeat
        print(f"Overall average accuracy for {k}-fold cross validation, {classifier} classifier with {kernel_type} kernel, C = {C}, arg(sigma/D)={arg}: {format((overall_accuracy*100), '.2f')}%")

    return overall_accuracy


# ## decision_boundary_plot
#
# decision_boundary_plot() accepts numpy arrays of data and a classifier function and plots the decision boundary of the classifier on the data.  Data must be 2-dimensional, and is currently 
# written to accept a maximum of 4 classes. The data should be split into an X and Y array with X being observations and Y being labels.  The classifier function should accept the X and Y arrays as
# training data and return a numpy array of classifications for the test data, which will be the pixels in the decision boundary plot.
#
# **Parameters**
#
# * **data_x** np.ndarray: 2-dimensional array of observations
# * **data_y** np.ndarray: 1-dimensional array of labels
# * **classifier** callable: A function that accepts training data and returns classifications for test data.
# * **plot_info** dict: A dictionary of plot information such as title, xlabel, and ylabel.
#
# **returns** None
def decision_boundary_plot(data_x: np.ndarray, data_y: np.ndarray, classifier: callable, plot_info: dict = {}):

    classes = np.unique(data_y)

    padding = 0.5
    xmin = min(data_x[:,0]) - padding
    xmax = max(data_x[:,0]) + padding
    ymin = min(data_x[:,1]) - padding
    ymax = max(data_x[:,1]) + padding

    # Makes a grid of points to classify using np.meshgrid
    x_axis = np.linspace(xmin, xmax, 125)
    y_axis = np.linspace(ymin, ymax, 70)
    xx, yy = np.meshgrid(x_axis, y_axis)
    grid_points = np.vstack([xx.ravel(), yy.ravel()]).T

    grid_classifications = classifier(data_x, data_y, grid_points)

    bg_colors = ['lightsalmon','palegreen','lightblue', 'lightgray']
    fg_colors = ['darkred','darkgreen','darkblue', 'black']
    fg_colors2 = ['r','g','b', 'k']

    fig, ax = plt.subplots()
    ax.set(**plot_info, xlim = (xmin,xmax), ylim = (ymin,ymax))

    # Plots background grid points for each species
    for i,cls in enumerate(classes):

        ax.scatter(data_x[np.where(data_y == cls)][:,0], 
                    data_x[np.where(data_y == cls)][:,1], 
                    s=10, 
                    edgecolors=fg_colors[i], 
                    marker='o', 
                    facecolors = fg_colors[i], 
                    clip_on=False, 
                    label=cls)
        
        # plots background grid points
        ax.plot(grid_points[np.where(grid_classifications == cls)][:,0], 
                grid_points[np.where(grid_classifications == cls)][:,1],
                marker='o', 
                color=bg_colors[i], 
                linestyle='none', 
                markersize=5,
                zorder= -100)
        
        # plots decision boundary line
        countour_heights = np.where(grid_classifications == cls, 1, 0)
        ax.contour(xx, yy, countour_heights.reshape(xx.shape),
                    levels=1,
                    colors='black', 
                    linestyles='solid', 
                    linewidths=2,
                    zorder=1)
        
        ax.legend(loc='upper left')

    plt.show()