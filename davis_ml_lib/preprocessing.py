# Custom helper functions
from . import helpers as hlp

# Third-party imports
import pandas as pd
import numpy as np
import seaborn as sns
from itertools import combinations
from sklearn.decomposition import PCA
from scipy.fft import dctn


#
##
#######################Feature Ranking/Selection#######################
##
#

####### 2D Fisher Discriminant Ratio #######

def get_2Dfdr_matrix(data: pd.DataFrame, class_indices: dict) -> pd.DataFrame:
    '''
    Accepts a pandas DataFrame and a dictionary of class indexes, and returns a pandas DataFrame 
    containing the 2D Fisher Discriminant Ratio (FDR) for each combination of features in the frame.

    The dataframe passed in must only include columns that should be included in the calculation (all values must be numerical).
    The class_indices dictionary should contain a list of indexes for each class in the dataframe, e.g. {class_1: [index list], class_2: [index list], ...}

    **Parameters**

    * **data** dataframe: the pandas dataframe with columns for which the 2D FDR will be calculated.
    * **class_indices** dict: the dictionary containing a list of indexes for each class in the dataframe.

    **returns** dataframe: The 2D FDR for each column in the input.  Retains column indexing.
    '''

    fdr_matrix = pd.DataFrame(columns=data.columns)

    for class_pair in combinations(class_indices, 2):
        #display(data.iloc[class_indices[class_pair[0]]])
        data1 = data.iloc[class_indices[class_pair[0]]]
        data2 = data.iloc[class_indices[class_pair[1]]]

        mean_vector = (data1.mean() - data2.mean()) ** 2
        std_vector = data1.std()**2 + data2.std()**2

        zero_index = np.where(std_vector == 0)[0]
        std_vector[zero_index] = 10000

        ratio = mean_vector / std_vector

        zero_index = np.where(abs(ratio) > 10000)[0]
        ratio[zero_index] = 0
        # ratio.name = f"{class_pair[0]}v{class_pair[1]}"
        fdr_matrix.loc[f"{class_pair[0]}v{class_pair[1]}"] = ratio
        # fdr_matrix = fdr_matrix.append(ratio)
       
    return fdr_matrix


####### Nearest-mean feature ranking #######
# Returns a ranked list of features and combinations of features based on
# their accuracy in a simple nearest-mean classification algorithm.

# data: A pandas dataframe containing the data to be classified
# features: A list of column names to be used as features
# category_column: The name of the column containing the category labels

# Note there are many better available techniques for feature selection.
# This was primarily a learning exercise.
def feature_rank(data: pd.DataFrame, features: list, category_column: str):

    categories = data[category_column].unique()
    features = list(features)

    #Create blank DataFrame to store scores by feature combination
    output_scores = pd.DataFrame(columns=['accuracy'])

    #Iterate over each possible combination of features, from 1 to the total # of features
    for i in range(len(features)):
        for columns in combinations(features, i+1):

            columns = list(columns)

            #Setup for storing data
            classification_data = pd.DataFrame(columns=(categories))

            #Get Mahalanobis Distance scores for each data point relative to different category means & covariances            
            for category in categories:

                #Prepare category mean/covariance
                working_data = data[data[category_column] == category][columns]
                mean_vector = working_data.mean().to_numpy()
                cov_m = hlp.get_covariance_m(working_data)

                #Find M distance from all data points to local category mean
                classification_data[category] = data[columns].apply(hlp.m_distance, axis = 1, raw = True, args= (cov_m, mean_vector))

            #List which category's mean each data point has lowest M distance to
            classification_data['classification'] = classification_data.idxmin(axis=1)

            #Compare with given list of classes to give accuracy score by feature combination
            correct = count = 0
            for index in classification_data['classification'].index:
                if classification_data.loc[index, 'classification'] == data.loc[index,category_column]:
                    correct += 1
                count +=1
            accuracy = correct / count
            output_scores.loc[str(columns), 'accuracy'] = accuracy
    
    #Rank feature combinations in descending order
    output_scores = output_scores.sort_values(by='accuracy', ascending=False)
    return output_scores

#
##
##############Principle Component Analysis/Dimensionality Reduction########################    
##
#

# Performs PCA on a dataframe and returns a dataframe with the specified number of principal components
# Also plots data in 2D with the first two principal components if verbose is set to True
# 
# This particular function is not a custom one, but the MNIST_eigendecomp() function seen below is
# basically just a PCA function, and I haven't bothered to make a general PCA function yet.  I may
# in the future, but use this for now.
#
# If a pca model is provided, n_components is ignored.  Choosing # of components based on explained variance
# is currently not implemented.
def pca_n(data: np.ndarray, y: np.ndarray = None, *, pca_model = None, n_components: int = 2, verbose: bool = False, reuse: bool = False) -> pd.DataFrame:

    if pca_model is None:
        pca_model = PCA(n_components=n_components)
        data = pca_model.fit_transform(data)
    else:
        data = pca_model.transform(data)

    if verbose == True:
        print("Explained variance ratio: ",pca_model.explained_variance_ratio_, " Total explained: ",round(sum(pca_model.explained_variance_ratio_)*100, 2),"%", sep='')
        print("Explained variance:",pca_model.explained_variance_)
        if y is not None:
            sns.relplot(data=data[:2], x=data[:,0], y=data[:,1], hue=y)

    if reuse == False:
        return data
    else:
        return data, pca_model


#
##
################# Standardizing and normalizing dataframes and numpy arrays: ####################
##
#

def standardize_df(data: pd.DataFrame, reuse: bool = False) -> pd.DataFrame:
    '''
    Standardizes a dataframe to have mean 0 and standard deviation 1.
    If reuse is set to True, returns the standardized dataframe, 
    the mean, and the standard deviation (for reuse).

    data: pandas DataFrame 
    '''
    
    mean = data.mean()
    st_d = data.std()
    data = data.sub(mean).div(st_d)
    if reuse == False:
        return data
    else:
        return data, mean, st_d
    
def normalize_df(data: pd.DataFrame, reuse: bool = False) -> pd.DataFrame:
    min = data.min()
    max = data.max()
    data = data.sub(min).div(max - min)
    if reuse == False:
        return data
    else:
        return data, min, max
    
def standardize(data: np.ndarray, mean: float = None, st_d: float = None, reuse: bool = False) -> np.ndarray:

    if mean is None:
        mean = data.mean()
    if st_d is None:
        st_d = data.std()
    data = (data - mean) / st_d
    if reuse == False:
        return data
    else:
        return data, mean, st_d

def normalize(data: np.ndarray, min: float = None, max: float = None, reuse: bool = False) -> np.ndarray:
    
    if min is None:
        min = data.min()
    if max is None:
        max = data.max()
    data = (data - min) / (max - min)
    if reuse == False:
        return data
    else:
        return data, min, max
    


#
##
####################################################### MNIST ###############################################
##
#

# Mask matrices to extract diagonal, vertical, and horizontal components of 
# a DCT matrix.  Used in the DCT-based feature generation method below.
DIAG_MASK=np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                   [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                   [0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                   [0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                   [0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                   [0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                   [0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                   [0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                   [0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                   [0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0],
                   [0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0],
                   [0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0],
                   [0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0],
                   [0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                   [0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                   [0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                   [0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                   [0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                   [0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                   [0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                   [0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                   [0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                   [0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                   [0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                   [0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                   [0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                   [0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                   [0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
                  ])

VERT_MASK =np.array([[0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                  [0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                  [0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                  [0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                  [0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                  [0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                  [0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                  [0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                  [0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1],
                  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1],
                  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1],
                  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1],
                  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
                  ])

HORIZ_MASK = np.array([[ 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])

MASKS = [DIAG_MASK, VERT_MASK, HORIZ_MASK]

######### MNIST Preprocessing #########

def mnist_dct(image: np.ndarray, masks: list = None) -> list:
    """
    Accepts an nxn image and returns the DCT of the image, separated by input
    masks if provided.  Intended for use on MNIST dataset with masks separating
    into horizontal, vertical, and diagonal components.  Output components are 
    flattened and returned in a list, or as a single array if no masks provided.
    Masks must be the same shape as images.

    image: nxn numpy array
    masks: list of nxn numpy arrays

    returns: list of 1D numpy arrays, one for each mask
    """

    dct_data = dctn(image, norm='ortho')

    if masks is None:
        return np.reshape(dct_data, -1)
    
    else:
        output = []
        for mask in masks:
            output.append(dct_data[mask==1])
        return np.array(output, dtype=object)



def mnist_eigendecomp(data: np.ndarray, n:int = 20, reuse: bool = False, eigendata: list = None) -> np.ndarray:
    """
    Accepts an array of image data separated into DCT components, generates and
    selects the top n features of each component via eigendecomposition, and returns
    the generated data along with the size-n eigenvector matrix for each component if
    marked for reuse.

    data: 3D numpy array with shape (num_values, num_components)
    n: number of features to select for each component
    reuse: if True, will return the top n eigenvectors along with the transformed data
    eigendata: if not None, will use provided, previously calculated eigenvectors

    returns: 
    """
    if eigendata == None:
        get_new_eigendata = True
        eigendata = []
    else:
        get_new_eigendata = False

    num_values = data.shape[0]
    num_components = data.shape[1]
    data = np.transpose(data)

    output = []
    
    for i in range(num_components):

        tmp = np.stack(data[i], axis=0)
        
        if get_new_eigendata:
            cov = np.cov(tmp - tmp.mean(axis=0), rowvar=False)
            eigvals, eigvecs = np.linalg.eig(cov)
            eigendata.append(eigvecs[:, :n])
        
        output.append(tmp @ eigendata[i])

    output = np.stack(output, axis=1)
    output = output.reshape(num_values, -1)

    if reuse:
        return output, eigendata
    else:
        return output
    

def mnist_preprocess(data: np.ndarray, *,
                     preproc_model: dict = None, 
                     masks: list = MASKS, 
                     dct: bool = True, 
                     num_features:int = 60, 
                     max_std: float = 3.0, 
                     normalization: str = None,
                     pca_components: int = None,
                     reuse = False) -> np.ndarray:
    """
    Preprocesses MNIST data with a series of processing functions.
    If a processing model is provided, the parameters from the model
    will be used; any other provided arguments will be ignored.

    :data: MNIST data. Must be a 3D array of shape (num_images, 28, 28) (other size images
           can be used, but must be square, and DCT masks must be resized accordingly.  This
           has not been tested with images other than the MNIST dataset, so YMMV.)

    :processing_model: a dictionary with the following possible entries:
    {
        'masks': list of masks,
        'dct': whether to apply dct,
        'num_features': number of features to keep (if dct is True),
            'dct_eigendata': (num_features) eigenvectors from the eigendecomposition 
                             of the dct matrix.
        'max_std': maximum standard deviation (DEPRACATED)
        'normalization': normalization method
            'min', 'max': min and max values for normalization
            'mean', 'std': mean and std values for standardization
    }
    :dct: whether to apply dct
    :num_features: number of features to keep (if dct is True)
                   num_features will be divided by num_masks (rounded down), and equal numbers of features
                   will be kept from each of the individual components after eigendecomposition.
    :max_std: maximum standard deviation
    :normalization: normalization method, can be 'minmax' (all data in [0,1]) or 'standard' (mean 0, std 1)
    :pca_components: number of components to keep from PCA decomposition. If None, PCA is not applied.
    :return: preprocessed MNIST data, with a 'model' dictionary if reuse = True
    """

    if preproc_model is not None:
        masks = preproc_model['masks']
        dct = preproc_model['dct']
        num_features = preproc_model['num_features']
        max_std = preproc_model['max_std']
        normalization = preproc_model['normalization']
        pca_components = preproc_model['pca_components']
    else:
        preproc_model = {}
        preproc_model['masks'] = masks
        preproc_model['dct'] = dct
        preproc_model['num_features'] = num_features
        preproc_model['max_std'] = max_std
        preproc_model['normalization'] = normalization
        preproc_model['pca_components'] = pca_components
    
    # This is MNIST specific.  Change if using differently-sized images.
    data = np.reshape(data, (-1, 28, 28))

    # Apply Discrete Cosine Transform, separate into horizontal, vertical, and diagonal components, 
    # and keep only the first (num_features // num_masks) features from each component
    if dct:

        data = np.array([mnist_dct(image, masks=masks) for image in data])

        if 'dct_eigendata' in preproc_model.keys():
            dct_eigendata = preproc_model['dct_eigendata']
        else:
            dct_eigendata = None

        data, dct_eigendata = mnist_eigendecomp(data, n=(num_features // len(masks)), reuse=True, eigendata=dct_eigendata)
        preproc_model['dct_eigendata'] = dct_eigendata

    # Remove outliers based on a maximum 'standard deviation' equivalent (DEPRACATED)
    # A note:  I previously implemented a method of removing outliers using Mahalanobis distance, removing the equivalent
    # of 'n' standard deviations from the mean, translated to the correct Mahalanobis distance from the mean based on dimensionality.  
    # I have since learned that this method works poorly for high-dimensional data, especially when the data may not be normally distributed.
    # So, I just took it out.  I may reimplement later with better techniques.

    # Normalize data
    if normalization is not None:
        if normalization == 'minmax':
            if 'min' in preproc_model.keys() and 'max' in preproc_model.keys():
                min = preproc_model['min']
                max = preproc_model['max']
                data = normalize(data, min=min, max=max)
            else:
                data, min, max = normalize(data, reuse=True)
                preproc_model['min'] = min
                preproc_model['max'] = max
        elif normalization == 'standard':
            if 'mean' in preproc_model.keys() and 'std' in preproc_model.keys():
                mean = preproc_model['mean']
                std = preproc_model['std']
                data = standardize(data, mean=mean, std=std)
            else:
                data, mean, std = standardize(data, reuse=True)
                preproc_model['mean'] = mean
                preproc_model['std'] = std
        else:
            raise ValueError('Normalization method must be "minmax" or "standard" or None.')
        
    # Apply PCA
    if pca_components is not None:
        if 'pca_model' in preproc_model.keys():
            pca_model = preproc_model['pca_model']
        else:
            pca_model = None
        data, pca_model = pca_n(data, n=pca_components, pca_model=pca_model, reuse=True, verbose = True)
        preproc_model['pca_model'] = pca_model
    
    if reuse:
        return data, preproc_model
    else:
        return data
        