# v20.2
# Version notes
# - given the repeated warnings from rdkit regarding the future decommissioning of Morgan fingerprint calculations,
#   all occurrences of such calculations were replaced by the 'new' method that uses a generator.
# - the listing of fingerprint bits is now always done using GetOnBits(), given that all new fingerprints are bit, not integers,
#   even when they are unfolded. Hopefully this will not impair the functioning of the code.

# 0. LOADING OF THE REQUIRED PACKAGES AND MODULES

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit import DataStructs
import rdkit
from collections import defaultdict
import pandas as pd
from rdkit.Chem import rdMolDescriptors
import matplotlib.pyplot as plt
import numpy as np
from rdkit.SimDivFilters import rdSimDivPickers
import pulp
from pulp import *
import gzip
import math
from scipy.stats import norm

# 1. DEFINITION OF THE REQUIRED FUNCTIONS

# 1.0.1. CSV file chunk reader (to save memory when a csv has many sparse columns)

def read_csv_to_dataframe_with_sparse_cols(
    input_csv_file_full_path,
    dense_columns_names = []):
    """
    Reads a csv file to a sparse pandas dataframe.
    Standard pandas.read_csv() can generate extremely large dataframes when the csv is sparse, sometimes crashing the memory.
    
    Parameters
    ----------
    input_csv_file_full_path : str
        the path to the *csv* file to read (can be (g)zipped)
    dense_column_names : list of strings, or empty list
        the list of column names, in the input file, that are completely filled (not sparse)
        - these columns will be read by standard pandas.read_csv(), which is faster
        
    Returns
    -------
    pandas.DataFrame with the relevant columns set to pandas.arrays.SparseArray type
    """
    # Read first the column names
    firstrow = pd.read_csv(input_csv_file_full_path, nrows = 1)

    # Identify the data columns (all those that are not the compound ID, not the Smiles)
    cols_with_data = firstrow.columns.values[~(firstrow.columns.isin(dense_columns_names))]

    # Make a DataFrame with dense columns, if any
    if (len(dense_columns_names) != 0):
        df = pd.read_csv(input_csv_file_full_path, usecols = dense_columns_names)
        total_N_rows_to_read = df.shape[0]
    else:
        df = pd.DataFrame()
        dfr = pd.read_csv(input_csv_file_full_path, usecols = [firstrow.columns.values[0]])
        total_N_rows_to_read = dfr.shape[0]

    # Then read in the data columns in chunks of 10^8 cells, adding them to df as sparse
    print("Reading the data columns from the csv file...")
    print("")
    N_cols_left_to_read = len(cols_with_data)
    N_cols_to_read_each_chunk = np.ceil(100000000 / total_N_rows_to_read).astype('int32')

    index0 = 0

    while (N_cols_left_to_read > 0):
        #print("N_cols_left_to_read = ", N_cols_left_to_read)
        #print("dimensions of df =", str(df.shape))
        next_N_cols_to_read = min(N_cols_left_to_read, N_cols_to_read_each_chunk)
        current_cols_to_read = list(cols_with_data[index0:(index0 + next_N_cols_to_read)])
        data = pd.read_csv(input_csv_file_full_path, usecols = current_cols_to_read)
        for i in range(next_N_cols_to_read):
            data[current_cols_to_read[i]] = pd.arrays.SparseArray(data[current_cols_to_read[i]])    
        df = pd.concat([df, data], axis = 1)
        index0 += next_N_cols_to_read
        N_cols_left_to_read -= next_N_cols_to_read

    return df

# 1.0.2. CSV file chunk writer (to speed up writing dataframes that have sparse columns)

def write_csv_from_dataframe_with_sparse_cols(
    dataframe,
    sparse_columns_names,
    output_csv_file_full_path):
    """
    Writes a csv file from a pandas.DataFrame that has some columns in pandas.arrays.SparseArray type,
    by splitting it into chunks of rows and making them dense before writing to file, in 'append' mode.
    At the time of writing, using df.to_csv() on such a DataFrame would be extremely slow, hence the need for this function.
    
    Parameters
    ----------
    dataframe : pandas.DataFrame object
        the DataFrame to write to file
    sparse_column_names : list of strings
        the names of the columns that are of pandas.arrays.SparseArray type in dataframe
        - this is required and cannot be empty (if all columns are dense, to_csv() works better than this function)
    output_csv_file_full_path
        the path to the *csv* file to write (can be (g)zipped)
        - NOTE: if the file already exists, it will be deleted/overwritten!
        
    Returns
    -------
    None.
    The output file is silently written out.
    """
    # remove output file if it exists, as it will be written in 'append' mode
    import os
    if os.path.exists(output_csv_file_full_path):
        os.remove(output_csv_file_full_path)

    # calculate how many chunks need to be made so that at each iteration ~10^8 cells are read
    N_chunks = np.ceil(dataframe.shape[0] * dataframe.shape[1] / 100000000).astype('int32')

    # the native pandas .to_csv function works well for this case, using mode = 'a',
    # and handles automatically all the compression part      
    for i, chunk in enumerate(np.array_split(dataframe, N_chunks)):
        #print("Chunk ",str(i+1), " / ", N_chunks)
        chunk[sparse_columns_names] = chunk[sparse_columns_names].sparse.to_dense()
        #chunk.to_csv(out, header=i==0, index = False)
        chunk.to_csv(output_csv_file_full_path, mode = 'a', header=i==0, index = False)
                    
# 1.0.3. Unpivoting function, especially useful for sparse data

def unpivot_dataframe(
    dataframe,
    ID_column_name,
    data_columns_names,
    property_column_name,
    value_column_name = None):
    """
    Takes a 'wide' (= pivoted) pandas.DataFrame, and converts it to a 'long' (= unpivoted) pandas.DataFrame.
    > this is ~equivalent to pandas.melt(). Differences:
      - at the time of writing, a DataFrame with columns in pandas.arrays.SparseArray type raises an error when using melt(), hence the need for this function
      - NOTE: the records corresponding to the ID column are reported at the beginning, with property_column_name set to ID_column_name,
        and value set to 1 (this is specific to the data balancing workflow), of length equal to the number of rows of the 
    
    Parameters
    ----------
    dataframe : pandas.DataFrame object
        the 'wide' (= pivoted) data frame to unpivot
    ID_column_name : string
        the name of the *strictly dense* column that identifies distinct records (ideally this column should only have unique values, but it's not mandatory)
    data_columns_names : list of strings
        the names of the columns that contain data
        - in principle this may just be [c for c dataframe.columns if c := ID_column_name]
        - but sometimes not all the non-ID columns in dataframe are useful; this is a chance to get rid of them
    property_column_name : string
        name of the column, in the output 'long' pandas.DataFrame, with values corresponding to the names from data_columns_names in dataframe
    value_column_name : string or None
        optional name of the column, in the output 'long' pandas.DataFrame, containing the *values* of the data_columns_names in dataframe
        - if None, the output will not contain the former data columns values, but only their names, repeated as many times as per input
    
    Returns
    -------
    A 'long' (= pivoted) pandas.DataFrame with strictly dense columns [ID_column_name, property_column_name, (value_column_name)]
        
    Example
    -------    
    - input dataframe:
    
    cluster    A       B       C
    1         0.5     0.8     nan
    1         nan     1.1     nan
    2         "x"     nan     "g"
    2         0.4     0.7     "y"
    3         nan     nan     0.1    
    - parameters used:
    
    ID_column_name = "cluster", data_columns_names = ["A","B","C"], property_column_name = "task_name", value_column_name = "task_value"
    
    - output dataframe
    cluster    task_name   task_value
    1          "cluster"        1
    1          "cluster"        1
    2          "cluster"        1
    2          "cluster"        1
    3          "cluster"        1
    1          "A"              0.5
    2          "A"              "x"
    2          "A"              0.4
    1          "B"              0.8
    1          "B"              1.1
    2          "B"              0.7
    2          "C"              "g"
    2          "C"              "y"
    3          "C"              0.1
    """
    ids = dataframe[ID_column_name]
    data_ids = list(dataframe[ID_column_name])
    task_names = list(np.repeat(ID_column_name, len(data_ids)))
    values = list(np.repeat(1, len(data_ids)))

    for c in data_columns_names:
        c_data = dataframe[c]
        c_data_notna_bool = c_data.notna()
        c_indices = c_data.index[c_data_notna_bool]
        data_ids.extend(ids[c_indices])
        c_N_data = len(c_indices)
        task_names.extend(np.repeat(c, c_N_data))
        if (value_column_name is not None):
            values.extend(c_data[c_data_notna_bool])

    if (value_column_name is None):
        df_dataonly_unpivoted = pd.DataFrame({ID_column_name : data_ids,
                                        property_column_name : task_names} )
    else:
        df_dataonly_unpivoted = pd.DataFrame({ID_column_name : data_ids,
                                        property_column_name : task_names,
                                        value_column_name : values })

    return df_dataonly_unpivoted

# 1.1. Top level function

def make_chemically_disjoint_data_balanced_ML_subsets(
    path_to_input_csv,
    path_to_output_csv,
    sizes,
    equal_weight_perc_compounds_as_tasks = False,
    balance_categorical_labels = False,
    balance_continuous_distributions = False,    
    interpret_censored_floats_as_floats = True,
    N_bins = 5,
    folded_to_BitVect = False,
    morgan_radius = 3,
    morgan_NBits = 32768,
    initial_clustering_method = "sphere exclusion",
    automated_search_of_best_initial_clustering = False,
    criterion_for_automated_search = "global intercluster distances",
    reduced_NBits_for_optim = 512,
    smiles_column_name = "SMILES",
    unique_compound_ID_column_name = None,
    initial_cluster_column_name = "initial_cluster",
    ML_subset_column_name = "ML_subset",
    calculate_min_inter_subset_TanimotoDists = False,
    min_is_dist_column_name = "min_inter_subset_Tanimoto_distance",
    save_data_summaries = True,
    N_centers_to_pick = None,
    min_dist = None,
    assign_to_initial_centers_only = True,
    similarity_aggregation = 'mean',
    seed = -1,
    priority_to_removal_of_bits_from_S0 = False,
    min_overlap = 4,
    min_sim = 0,
    relative_gap = 0.5,
    time_limit_seconds = 60,
    max_N_threads = 4):
    """
    Takes as input a *pivoted* csv file with SMILES, unique identifiers (optional), and tasks (properties to model) data, possibly sparse.
    Clusters the records into 'chemically homogeneous' groups (aiming to keep similar molecules together and different molecule separate), based on the SMILES.
    Re-merges clusters into the required number of final subsets, at the same time 'balancing' the data, i.e.:
      > in general: the number of records (=distinct SMILES) should be similar among all final subsets
      > for all types of data: for each task, the number of data points should be similar among all final subsets
      > for continuous data: for each task, the distributions of values should be similar among all final subsets
      > for categorical data: for each task, the % of data belonging each category should be similar among all final subsets
      
    Parameters
    ----------
    path_to_input_csv : string
        path to *pivoted* csv file with SMILES, unique identifiers (optional), and tasks (properties to model) data, possibly sparse.
        - NOTE: please make sure that no other unwanted columns are present in the file! Anything that is not SMILES or identifier will be treated as data!        
    path_to_output_csv : string
        path to *pivoted* csv output file
        - this will be the input file with the additional columns explained below (overwriting if they exist in the input)
    initial_cluster_column_name : string
        the initial cluster each record was assigned to.
        NOTE: if N clusters are made, the values of initial_cluster_column_name are 0 .. (n-1)
    ML_subset_column_name : string
        the name of the column, in the output csv, containing the index of the final ML subset each record was assigned to.
        NOTE: if sizes has length n, the values of ML_subset_column_name are 1 .. n
    min_is_dist_column_name : string        
        the minimal Tanimoto distance of the compound to any compound in any other ML subsets (if calculate_min_inter_subset_TanimotoDists == True).
    sizes : list of numbers
        list of the desired final sizes (will be normalised to fractions internally)    
    equal_weight_perc_compounds_as_tasks : bool
        - if True, matching the % records will have the same weight as matching the % data of individual tasks
        - if False, matching the % records will have a weight X times as large as matching the % data of the X tasks
    balance_categorical_labels : bool
        - if True, any column that has even just one non-'float-able' value will be considered categorical,
          > i.e. each distinct value will be considered a category (label), and balanced across the ML subsets.
          > E.g. if a column contains 'good' and 'bad' labels, in 30:70 proportions in the overall dataset,
            it will be attempted to have 30:70 good:bad also in all ML subsets (not only the correct # of data points).
        - Caution: if you have a column with many continuous values and even just one text value included by mistake,
          all distinct numerical values will be seen as a separate category, and probably crash the LP solver.
        - NOTE: sometimes binary categories are encoded as integers, like 0 and 1.
          > The 'correct' option would be to turn them into non-float-able text, like 'inactive' and 'active'.
          > An easier way around is to set balance_continuous_distributions = True with N_bins = 2. Same final effect.
    balance_continuous_distributions : bool
        - if True, any column that has *all* 'float-able' values will be binned into the desired N_bins (below),
          and it will be attempted to reproduce the same distribution in all ML subsets.
          > E.g. if a property 'Y' has bins (-5,-4], (-4,-3], ..., (4, 5], with data in proportions 3:7:...:4,
            it will be attempted to have the same bins and proportions in all ML_subsets.
    interpret_censored_floats_as_floats : bool
        - if True, values like '<5' or '>100' will be intepreted as floats (5 and 100) *for distribution balancing purposes*
          > meaning: the original data will still be saved as they are in the input, but the distribution calculations will use uncensored data
          > in addition, for convenience, an output file with uncensored data will be saved, too (only for columns that become fully continuous by uncensoring)
        - if False, such values will be interpreted as text (and the presence of even just one of them will make a column categorical!)
    N_bins : positive integer
        - only applicable if balance_continuous_distributions == True
        - the number of bins to make for the distribution of continuous variables to be balanced across subsets
        - Caution: 5 bins means that the LP variables to solve for are ~multiplied by 5.
          This may result in a *much* longer run time. Use only if strictly necessary / if you have seen distribution unbalance.
        - Technical note on binning:
          > pandas.cut is used, with ~equally sized intervals in the found range of continuous data.
          > Exception: when there are left or right outliers (<= Q1 - 1.5 * IQR or >= Q3 - 1.5 * IQR), those are not
            included in the calculation of the intervals, and they are added back in as lower and upper limits.
          > E.g. [-100, -1, 0, 1, 3, 9], request 5 bins --> -100 is outlying --> range 9-(-1) = 10 --> intervals of 10/5 = 2 units
            --> [-1, 1], (1, 3], (3, 5], (5, 7], (7, 9] --> [-100, 1], (1, 3], (3, 5], (5, 7], (7, 9]
    folded_to_BitVect : bool
        must the fingerprints be calculated as folded bit vectors?
        - if True, the calculations will be faster but the results less accurate
        - if False, full fingerprints will be used, yielding more accurate results
    morgan_radius : integer {1, 2, 3}
        the radius for the morgan fingerprint calculation, where applicable
    morgan_NBits : positive integer
        the number of bits for the folded fingerprints (only applicable when folded_to_BitVect == True)
    initial_clustering_method : string {"sphere exclusion", "iterative min overlap"}
        the method used to do the initial clustering;
        - "sphere exclusion" is faster and only requires one parameter (the minimal MaxMin distance)
        - "iterative min overlap" is slower, but generally achieves a better chemical separation
    automated_search_of_best_initial_clustering : bool
        - if True, you need to specify some boundaries (see below), and the function will look for
          the best clustering within those boundaries, by repeating it several times
            > Boundaries to specify for the automated search:
              - if initial_clustering_method == "sphere exclusion"
                --> specify parameter 'min_dist' as a *list* of 2 values, e.g. [0.6, 0.9]
                --> N_centers_to_pick will be ignored (or it would take priority over min_dist)
              - if initial_clustering_method == "iterative min overlap"
                --> specify parameter 'min_sim' as a *list* of 2 values, e.g. [0.5, 0.7]
          - if False, instead of boundaries you can *optionally* specify a single value for the
            parameters of the clustering method, which will be used to do a single initial clustering;
            default values will be applied if you do not specify any.
    criterion_for_automated_search : string {"global intercluster distances", "Shannon entropy"}
        - "global intercluster distances" is considerably slower, but gives a better result
        - "Shannon entropy" is much faster but only gives an approximate result
    reduced_NBits_for_optim : positive integer or None
        - if None, the search for the best clustering will be done using the fingerprints calculated as per above parameters
        - if numeric, the search for the best clustering will be done using folded fingerprints to this specified number of bits,
          mostly for speed, and once the best parameter is found, the clustering is rerun with the 'full' fingerprints
    smiles_column_name : string
        name of the column containing the SMILES in the input csv
    unique_compound_ID_column_name : string (optional) or None
        name of the column containing unique compound ID's in the input
    calculate_min_inter_subset_TanimotoDists : bool
        must inter-subsets distances be measured?
        - NOTE 1: this calculation requires a very long time, so do it only if strictly necessary
        - NOTE 2: the calculation is run on the final ML subsets, not on intermediate clusters
    save_data_summaries : bool
        must summaries of data counts, % and dists be saved?
        - if True, an output file with the counts, and one with % of data will be saved,
          replacing '.csv' by '_data_counts.csv', or '_data_percentages.csv'
        - if unique_compound_ID_column_name == None, the column in the summaries that refer to the
          count and % of *records* (not data) will be labelled '_temp_ID_column_name_mcddbMs'
          (please ensure that there are no pre-existing columns with this name in the input file)
    > Parameters that are only applicable when initial_clustering_method == "sphere exclusion":
        N_centers_to_pick : integer >= len(sizes)
        min_dist : float between 0 and 1
        assign_to_initial_centers_only : bool
        similarity_aggregation : string {'max', 'mean', 'median'}
        seed : integer
        - see documentation of function 'sphere_exclusion_clustering' for details    
    > Parameters that are only applicable when initial_clustering_method == "iterative min overlap":
        priority_to_removal_of_bits_from_S0 : bool
        min_overlap : integer >= 0
        min_sim : float between 0 and 1
        - see documentation of function 'iterative_clustering_by_minimal_overlap' for details
    > Parameters of the linear programming solver
        relative_gap
        time_limit_seconds
        max_N_threads
        - see documentation of function 'balance_data_from_tasks_vs_clusters_array_pulp' for details
    
    Returns
    -------
    A list with:
    0. number of initial clusters
    1. list of per-ML_subset median of inter-subset min Tanimoto distances
    2. the min of the list in 1
    3. weighted sum of absolute differences: given the (S x M) matrix P of data fractions by task (columns) vs ML_subset (rows),
       and given the (S x 1) column vector fsz from fractional_sizes = sizes / np.sum(sizes),
       and given the (1 x S) row vector skh from sk_harmonic = (1 / fractional_sizes) / np.sum(1 / fractional_sizes)
       --> sum(skh.abs(P - fsz))
       > essentially a measure of the distance of the solution from the 'ideal' case where each task has the exact required
         % of data points in each ML subset
    4. pandas DataFrame with the output of the automated clustering optimiser (empty if no automated clustering was done)
        
    Saves:
    - an intermediate output csv file with the initial cluster, to avoid losing it
      in case the process stops later on;
      File name: path_to_input_csv with '.csv' replaced by '_clustered.csv'
    - two intermediate output csv files with the data summaries per task, per initial cluster
      File name 1: path_to_input_csv with '.csv' replaced by '_clustered_data_counts.csv'
      File name 2: path_to_input_csv with '.csv' replaced by '_clustered_data_percentages.csv'
    - a final output csv file with the same data as the input file, plus columns with:
      - the identifier of the initial cluster
      - the identifier of the final ML subset 
      - optionally, the inter-subset distances
      File name : path_to_output_csv
    - if interpret_censored_floats_as_floats == True:
      > a file identical to path_to_output_csv BUT with censored data uncensored (e.g. with '>5' turned to 5)
        (only for columns that become fully continuous when uncensoring is applied)
        File name : path_to_output_csv with '.csv' replaced by '_uncensored.csv'    
    
    Optionally saves data summary files, to document the data balancing performance.    
    """
    # START PROCESSING
    
    # Check first if the required parameters are specified
    
    if (automated_search_of_best_initial_clustering == True):
        if (initial_clustering_method == "sphere exclusion"):
            if (type(min_dist) is list):
                pass
            else:
                raise ValueError("You selected the automated search and sphere exclusion clustering. You need to specify min_dist as a list of 2 values between 0 and 1.")
            if (len(min_dist) == 2):
                pass
            else:
                raise ValueError("You selected the automated search and sphere exclusion clustering. You need to specify min_dist as a list of 2 values between 0 and 1.")
            if ((min_dist[0] >= 0) & (min_dist[1] <= 1) & (min_dist[0] < min_dist[1])):
                pass
            else:                
                raise ValueError("You selected the automated search and sphere exclusion clustering. You need to specify min_dist as a list of 2 *distinct* values between 0 and 1, the first being smaller than the second.")
        elif (initial_clustering_method == "iterative min overlap"):            
            if (type(min_sim) is list):
                pass
            else:
                raise ValueError("You selected the automated search and iterative min overlap clustering. You need to specify min_sim as a list of 2 values between 0 and 1.")
            if (len(min_sim) == 2):
                pass
            else:
                raise ValueError("You selected the automated search and iterative min overlap clustering. You need to specify min_sim as a list of 2 values between 0 and 1.")
            if ((min_sim[0] >= 0) & (min_sim[1] <= 1) & (min_sim[0] < min_sim[1])):
                pass
            else:
                raise ValueError("You selected the automated search and iterative min overlap clustering. You need to specify min_sim as a list of 2 *distinct* values between 0 and 1, the first being smaller than the second.")
        else:
            raise ValueError("initial_clustering_method can only be either 'sphere exclusion' or 'iterative min overlap'.")
    else:
        if (initial_clustering_method == "sphere exclusion"):
            if (min_dist is None):
                if (N_centers_to_pick is None):
                    N_centers_to_pick = 10 * len(sizes)
                else:
                    if (N_centers_to_pick < len(sizes)):
                        raise ValueError("You cannot make the required number of final ML subsets by making a smaller number of initial clusters. Review your parameters.")
            else:
                if ((N_centers_to_pick is None) & (type(min_dist) is list)):
                    raise ValueError("min_dist is a list, although you required to do a direct clustering. \
                        Please specify either a single value for min_dist or a single value for N_centers_to_pick.")
    
    # BEGIN PROCESSING
    
    print("Split by clustering followed by remerge")
    print("=======================================")
    print("")
    
    # 1. Read data in csv format (file with smiles)
    
    print("Reading the compound ID's and SMILES from the csv file...")
    print("")
    # debug: this made a very large df, which can instead be made sparse
    #df = pd.read_csv(path_to_input_csv)
    # Use instead a function that reads columns in chunks and makes a dataframe with sparse columns
    
    # New in v19 : we allow unique_compound_ID_column_name to be None
    # To avoid errors when encountering duplicated ID's, and to allow users to NOT have an ID column in the input,
    # we now replace unique_compound_ID_column_name with the row numbers of df, keeping a matching dictionary if an ID was defined

    if unique_compound_ID_column_name == None :
        df = read_csv_to_dataframe_with_sparse_cols(
            dense_columns_names = [smiles_column_name],
            input_csv_file_full_path = path_to_input_csv)
        unique_compound_ID_column_name = '_temp_ID_column_name_mcddbMs'        
    else :
        df = read_csv_to_dataframe_with_sparse_cols(
            dense_columns_names = [unique_compound_ID_column_name, smiles_column_name],
            input_csv_file_full_path = path_to_input_csv)
        ID_vs_index_dict = dict(zip(list(range(df.shape[0])), list(df[unique_compound_ID_column_name])))
    df[unique_compound_ID_column_name] = list(range(df.shape[0]))
    
    # Check if all ID's are unique, halt the process if not
    # New in v19 : no longer needed, as now we are sure they are unique
    #if (len(np.unique(df[unique_compound_ID_column_name])) != len(df[unique_compound_ID_column_name])):
    #    raise ValueError("There are duplicates of '", unique_compound_ID_column_name,"' in the data. This is not foreseen. Data must be pivoted by unique ID.")
    
    cols_with_data = df.columns.values[~(df.columns.isin([unique_compound_ID_column_name, smiles_column_name]))]

    # 2,3. Convert the list of smiles to a list of fingerprints without storing molecules.
    print("Creating fingerprints from SMILES...")
    print("")

    fps = []
    fps_reduced = []

    # New in v20.2 : we use a Morgan FP generator, which must be initialised here
    if (folded_to_BitVect == True) :
        mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius = morgan_radius, fpSize = morgan_NBits)
    else :
        mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius = morgan_radius)
        # Not necessary to get the Bit Info Mapping, if we assume that GetOnBits() is consistent for both types of FP.
        #ao = rdFingerprintGenerator.AdditionalOutput()
        #ao.AllocateBitInfoMap()
    if ((automated_search_of_best_initial_clustering == True) & (reduced_NBits_for_optim is not None)):
        mfpgen_red = rdFingerprintGenerator.GetMorganGenerator(radius = morgan_radius, fpSize = reduced_NBits_for_optim)

    for s in df[smiles_column_name]:

        m = Chem.MolFromSmiles(s)

        if (folded_to_BitVect == True):
            #fps.append(rdMolDescriptors.GetMorganFingerprintAsBitVect(m, morgan_radius, morgan_NBits))
            fps.append(mfpgen.GetFingerprint(m))
        else:
            #fps.append(Chem.rdMolDescriptors.GetMorganFingerprint(m, morgan_radius, useCounts = False))
            fps.append(mfpgen.GetSparseFingerprint(m))

        if ((automated_search_of_best_initial_clustering == True) & (reduced_NBits_for_optim is not None)):            
            #fps_reduced.append(rdMolDescriptors.GetMorganFingerprintAsBitVect(m, morgan_radius, reduced_NBits_for_optim))
            fps_reduced.append(mfpgen_red.GetFingerprint(m))

    if ((automated_search_of_best_initial_clustering == True) & (reduced_NBits_for_optim is not None)):
        fps_for_optim = fps_reduced
    else:
        fps_for_optim = fps

    # 4. Make the clusters and final subsets
    
    print("Starting selections...")
    print("")
    
    # 4.1. Create the fractional sizes from the user-defined 'sizes' list
    fractional_sizes = sizes / np.sum(sizes)
    S = len(sizes)
    
    # 4.2. Make the initial clusters, either using the given parameters set, or by auto-search
    
    if (automated_search_of_best_initial_clustering == True):
        
        print("Started automated search of best initial clustering...")
        print("")
           
        if (initial_clustering_method == "sphere exclusion"):
            def cl_fun(md):
                cl = sphere_exclusion_clustering(list_of_molecules = None,
                        list_of_fingerprints = fps_for_optim,
                        N_centers_to_pick = None,
                        min_dist = md,
                        assign_to_initial_centers_only = assign_to_initial_centers_only,
                        similarity_aggregation = similarity_aggregation,
                        folded_to_BitVect = folded_to_BitVect,
                        morgan_radius = morgan_radius,
                        morgan_NBits = morgan_NBits,
                        seed = 5)
                if (criterion_for_automated_search == "Shannon entropy"):
                    out = min_intercluster_global_distances_and_clusters_Shannon_entropies(cl, fps_for_optim,  calculate_intercluster_dists = False)
                    median_Shannon_norm_entropy = np.median(out[3])
                    out2 = [median_Shannon_norm_entropy, cl]
                elif (criterion_for_automated_search == "global intercluster distances"):
                    out = min_intercluster_global_distances_and_clusters_Shannon_entropies(cl, fps_for_optim, calculate_intercluster_dists = True)
                    median_intercluster_global_Cosine_dist = np.median(out[1])
                    out2 = [median_intercluster_global_Cosine_dist, cl]
                else:
                    raise ValueError("criterion_for_automated_search can only be 'Shannon entropy' or 'global intercluster distances'.")
                return out2
            
            xl = min_dist[0]
            xr = min_dist[1]
                
        elif (initial_clustering_method == "iterative min overlap"):
            def cl_fun(msim):
                cl = iterative_clustering_by_minimal_overlap(list_of_molecules = None,
                       list_of_fingerprints = fps_for_optim,
                       priority_to_removal_of_bits_from_S0 = priority_to_removal_of_bits_from_S0,
                       min_overlap = min_overlap,
                       min_sim = msim,
                       folded_to_BitVect = folded_to_BitVect,
                       morgan_radius = morgan_radius,
                       morgan_NBits = morgan_NBits)[0]
                if (criterion_for_automated_search == "Shannon entropy"):
                    out = min_intercluster_global_distances_and_clusters_Shannon_entropies(cl, fps_for_optim,  calculate_intercluster_dists = False)
                    median_Shannon_norm_entropy = np.median(out[3])
                    out2 = [median_Shannon_norm_entropy, cl]
                elif (criterion_for_automated_search == "global intercluster distances"):
                    out = min_intercluster_global_distances_and_clusters_Shannon_entropies(cl, fps_for_optim, calculate_intercluster_dists = True)
                    median_intercluster_global_Cosine_dist = np.median(out[1])
                    out2 = [median_intercluster_global_Cosine_dist, cl]
                else:
                    raise ValueError("criterion_for_automated_search can only be 'Shannon entropy' or 'global intercluster distances'.")
                return out2
            
            xl = min_sim[0]
            xr = min_sim[1]
        
        min_dist_min = xl
        min_dist_max = xr
        
        delta = xr - xl
        tol = 0.01
        
        autosearch_criterion_endpoint_dict = dict()
        tried_clusterings_dict = dict()
        
        while (True):
    
            xm = xl + delta / 2

            print("xl,xm,xr = ",(xl,xm,xr))

            if xl not in autosearch_criterion_endpoint_dict:
                print("trying parameter =",xl)
                cl_out = cl_fun(xl)
                yl = cl_out[0]
                autosearch_criterion_endpoint_dict[xl] = yl
                tried_clusterings_dict[xl] = cl_out[1]
            else:
                yl = autosearch_criterion_endpoint_dict[xl]

            if xm not in autosearch_criterion_endpoint_dict:
                print("trying parameter =",xm)
                cl_out = cl_fun(xm)
                ym = cl_out[0]
                autosearch_criterion_endpoint_dict[xm] = ym
                tried_clusterings_dict[xm] = cl_out[1]
            else:
                ym = autosearch_criterion_endpoint_dict[xm]

            if xr not in autosearch_criterion_endpoint_dict:
                print("trying parameter =",xr)
                cl_out = cl_fun(xr)
                yr = cl_out[0]
                autosearch_criterion_endpoint_dict[xr] = yr
                tried_clusterings_dict[xr] = cl_out[1]
            else:
                yr = autosearch_criterion_endpoint_dict[xr]

            if (delta <= tol):
                max_d = max([yl,ym,yr])
                xbest = [xl,xm,xr][[yl,ym,yr].index(max_d)]
                break

            delta = delta / 2

            if ((yl <= ym) & (ym <= yr)):
                xr_new = min(min_dist_max, xr + delta)
                xl_new = xr_new - delta
                xl = xl_new
                xr = xr_new        
            elif ((yl >= ym) & (ym >= yr)):
                xl_new = max(min_dist_min, xl - delta)
                xr_new = xl_new + delta
                xl = xl_new
                xr = xr_new        
            else:
                xl = xm - delta / 2
                xr = xm + delta / 2

        xs = list(autosearch_criterion_endpoint_dict.keys())
        ys = list(autosearch_criterion_endpoint_dict.values())
        index_of_best = ys.index(max(ys))
        xbest = xs[index_of_best]
        max_d = ys[index_of_best]
        print("best parameter =",xbest, ", criterion output =", max_d)
        print("")
        
        ordered = [x for _,x in sorted(zip(xs, range(len(xs))))]
        xs_ordered = [xs[i] for i in ordered]
        ys_ordered = [ys[i] for i in ordered]
        plt.plot(xs_ordered, ys_ordered)
        plt.xlabel('trial parameter')
        plt.ylabel('criterion value to maximise')
        plt.show()
        xys = pd.DataFrame({'trial parameter' : xs_ordered, 'criterion value to maximise' : ys_ordered})
        print(xys)
        print("")

        cls = tried_clusterings_dict[xbest]
        
        if reduced_NBits_for_optim is not None:
    
            print("Redoing clustering with best parameter and full fingerprints...")
            print("")

            if (initial_clustering_method == "sphere exclusion"):
                cls = sphere_exclusion_clustering(list_of_molecules = None,
                        list_of_fingerprints = fps,
                        N_centers_to_pick = None,
                        min_dist = xbest,
                        assign_to_initial_centers_only = assign_to_initial_centers_only,
                        similarity_aggregation = similarity_aggregation,
                        folded_to_BitVect = folded_to_BitVect,
                        morgan_radius = morgan_radius,
                        morgan_NBits = morgan_NBits,
                        seed = seed)
            elif (initial_clustering_method == "iterative min overlap"):
                cls = iterative_clustering_by_minimal_overlap(list_of_molecules = None,
                       list_of_fingerprints = fps,
                       priority_to_removal_of_bits_from_S0 = priority_to_removal_of_bits_from_S0,
                       min_overlap = min_overlap,
                       min_sim = xbest,
                       folded_to_BitVect = folded_to_BitVect,
                       morgan_radius = morgan_radius,
                       morgan_NBits = morgan_NBits)[0]
            else:
                raise ValueError("initial_clustering_method can only be either 'sphere exclusion' or 'iterative min overlap'.")
        else:
            cls = tried_clusterings_dict[xbest]
        
    else:        
        print("Doing initial clustering by the defined parameters...")
        print("")
        
        xys = pd.DataFrame()

        if (initial_clustering_method == "sphere exclusion"):
            cls = sphere_exclusion_clustering(list_of_molecules = None,
                    list_of_fingerprints = fps,
                    N_centers_to_pick = N_centers_to_pick,
                    min_dist = min_dist,
                    assign_to_initial_centers_only = assign_to_initial_centers_only,
                    similarity_aggregation = similarity_aggregation,
                    folded_to_BitVect = folded_to_BitVect,
                    morgan_radius = morgan_radius,
                    morgan_NBits = morgan_NBits,
                    seed = seed)
        elif (initial_clustering_method == "iterative min overlap"):
            cls = iterative_clustering_by_minimal_overlap(list_of_molecules = None,
                   list_of_fingerprints = fps,
                   priority_to_removal_of_bits_from_S0 = priority_to_removal_of_bits_from_S0,
                   min_overlap = min_overlap,
                   min_sim = min_sim,
                   folded_to_BitVect = folded_to_BitVect,
                   morgan_radius = morgan_radius,
                   morgan_NBits = morgan_NBits)[0]
        else:
            raise ValueError("initial_clustering_method can only be either 'sphere exclusion' or 'iterative min overlap'.")
    
    # Store the initial cluster numbers in df
    df[initial_cluster_column_name] = cls

    path_to_intermediate_clustered_csv = path_to_input_csv.replace(".csv", "_clustered.csv")

    print("Saving intermediate clustered file in ",path_to_intermediate_clustered_csv," ...")
    print("")
    # The use of sparse DataFrame columns causes horrible slowness in to_csv.
    # To counteract that, the file is written out in chunks, using an appropriately defined function.
    #df.to_csv(path_to_intermediate_clustered_csv, index = False)
    
    # New in v19: if an ID was defined, hence dictionary ID_vs_index_dict exists, we need to map it back to df before saving
    # otherwise we drop the temporary column made from the row numbers
    df_temp = df.copy()
    try:
        df_temp[unique_compound_ID_column_name] = df_temp[unique_compound_ID_column_name].apply(lambda x : ID_vs_index_dict[x])
    except:
        df_temp.drop(columns = [unique_compound_ID_column_name], inplace = True)
    
    write_csv_from_dataframe_with_sparse_cols(
        dataframe = df_temp,
        sparse_columns_names = cols_with_data,
        output_csv_file_full_path = path_to_intermediate_clustered_csv)
    
    # cleanup
    del df_temp
    
    print("Done.")
    print("")
    
    # And count how many clusters were made
    Nc = len(np.unique(cls))
    
    # debug
    print("# clusters made =", Nc)
    print("")

    # Unpivot the data, for making the summary faster
    print("Preparing for cluster merging + balancing")
    print("")

    # New: now we allow to balance labels and/or distributions
    # --> this is achieved by splitting task names by label or bin
    # --> this requires specific weights for tasks, otherwise a task that has many labels or bins is over-weighted
    # In this case, the values must be retained in the unpivoted dataset, for later use

    if ((balance_categorical_labels == True) or (balance_continuous_distributions == True) or (interpret_censored_floats_as_floats == True)):
        value_column_name = "value_name"
        # Define a function that returns an uncensored float from a value, otherwise '', optionally trying uncensoring
        # NOTE: inf needs to be considered valid floats (NA isn't; but in unpivoted data as per output of unpivot_dataframe there should be no NA's)
        def float_unc(v, try_uncensoring):
            try:
                v = v.lstrip()
            except:
                pass
            u = ''
            try:
                u = float(v)
            except:
                if try_uncensoring == True:
                    try:
                        v = str(v)
                        uf = float(v[1:])
                        cf = v[0]
                        if ((cf == '<') | (cf == '>')):
                            u = uf
                    except:
                        pass
            return u
        # Define a function that takes a list of values and makes outlier-robust cuts for pd.cut
        def make_robust_cuts(s, N_bins):
            Q1 = np.quantile(s, 0.25)
            Q3 = np.quantile(s, 0.75)
            IQR = Q3 - Q1
            if IQR != 0:
                lol = Q1 - 1.5 * IQR
                uol = Q3 + 1.5 * IQR
                s_red = [n for n in s if ((n > lol) & (n < uol))]
            else:
                s_red = s
            r = max(s_red) - min(s_red)
            step = r / N_bins
            cuts = [min(s_red) + i * step for i in range(N_bins + 1)]
            cuts[0] = min(s)
            cuts[-1] = max(s)
            # if cuts only contains 1 unique value, pd.cut will return nan --> correct to 1 (= pd.cut will make 1 bin)
            if len(pd.unique(cuts)) == 1 :
                cuts = 1
            return cuts
    else:
        value_column_name = None

    print("   Unpivoting the data columns...")
    print("")
    df_dataonly_unpivoted = unpivot_dataframe(
        dataframe = df,
        ID_column_name = unique_compound_ID_column_name,
        data_columns_names = cols_with_data,
        property_column_name = "task_name",
        value_column_name = value_column_name)

    print("   Indexing the tasks...")
    print("")
    # Collect the index of task data for each task_name, and store in a dict    
    unique_task_names = pd.unique(df_dataonly_unpivoted["task_name"])
    index_vs_task_name_dict = dict()
    type_vs_task_name_dict = dict()
    for tn in unique_task_names:
        task_index = df_dataonly_unpivoted[df_dataonly_unpivoted['task_name'] == tn].index
        index_vs_task_name_dict[tn] = task_index
        # Provisionally set all tasks to categorical
        type_vs_task_name_dict[tn] = 'cat'

    # Create the required weights dictionary,
    # by identifying the number of bins of each task (categorical or continuous) when necessary
    # or setting to 1 as default when no balancing of labels or distributions is required
    weights_for_tasks = dict()

    # Identify which tasks are continuous, taking into account the setting of interpret_censored_floats_as_floats
    # - obviously this is only applicable if there are categorical labels or continuous distributions to balance
    if ((balance_categorical_labels == True) or (balance_continuous_distributions == True) or (interpret_censored_floats_as_floats == True)):
        print("   Classifying tasks into strictly continuous and categorical...")
        print("")        
    
    for tn in unique_task_names:
        weights_for_tasks[tn] = 1
        if tn != unique_compound_ID_column_name :
            # If balancing categorical labels and/or continuous distributions is required, split the 'task_name' accordingly
            if ((balance_categorical_labels == True) or (balance_continuous_distributions == True) or (interpret_censored_floats_as_floats == True)):
                # Find the values for the task
                task_index = index_vs_task_name_dict[tn]
                vals = df_dataonly_unpivoted.loc[task_index, value_column_name].copy()
                # Apply float_unc to the values
                vals_u = vals.apply(lambda v: float_unc(v, try_uncensoring = interpret_censored_floats_as_floats))
                # Decide if the values were all float-able
                vals_are_all_numerical = False
                if all(vals_u != ''):
                    vals_are_all_numerical = True
                    # If so, store the knowledge that this task is continuous into type_vs_task_name_dict                    
                    type_vs_task_name_dict[tn] = 'cont'
                    # If it was required to interpret censored values as floats, replace vals with vals_u
                    if interpret_censored_floats_as_floats == True:
                        #df_dataonly_unpivoted.loc[task_index, value_column_name] = list(vals_u.values)
                        vals = vals_u.copy()
                # For tasks that have at least one non-numerical value, if required to balance the labels,
                # rename task_name's according to the values
                if ((balance_categorical_labels == True) & (vals_are_all_numerical == False)) :                    
                    split_tasks = [str(tn) + '_(' + str(v) + ')' for v in vals]
                    df_dataonly_unpivoted.loc[task_index, 'task_name'] = split_tasks
                    unique_vals = pd.unique(vals)
                    split_weight = 1 / len(unique_vals)
                    for stn in pd.unique(split_tasks) :
                        weights_for_tasks[stn] = split_weight                    
                    print("    > Splitting categorical property '" + str(tn) + "' into " + str(len(unique_vals)) + ' columns...')
                # For tasks that have only numerical values, if required to balance data distributions,
                # bin the vals and rename task_name's according to the bins
                elif ((balance_continuous_distributions == True) & (vals_are_all_numerical == True)) :                    
                    cuts = make_robust_cuts(list(vals), N_bins)
                    vals_bins = pd.cut(vals, bins = cuts, include_lowest = True, duplicates = 'drop')
                    split_tasks = [str(tn) + '_' + str(b) for b in vals_bins]
                    df_dataonly_unpivoted.loc[task_index, 'task_name'] = split_tasks
                    unique_vals_bins = pd.unique(vals_bins)
                    split_weight = 1 / len(unique_vals_bins)
                    for stn in pd.unique(split_tasks) :
                        weights_for_tasks[stn] = split_weight                    
                    print("    > Splitting continuous property '" + str(tn) + "' into " + str(len(unique_vals_bins)) + ' columns...')

    print("   Making data summaries per task, per initial cluster...")
    print("")

    # Create the intermediate data summary before balancing
    # First, map the initial cluster numbers onto the pivoted data
    # NOTE to self: could be replaced by a dictionary method, but the ID should be unique in DF, so...
    df_dataonly_unpivoted = df_dataonly_unpivoted.merge(
        df[[unique_compound_ID_column_name,initial_cluster_column_name]],
        how = "left",
        left_on = unique_compound_ID_column_name,
        right_on = unique_compound_ID_column_name)

    # Then use cross tabulation to make the array of counts of data per cluster per task
    data_count_per_initial_cl = pd.crosstab(index = df_dataonly_unpivoted[initial_cluster_column_name], columns = df_dataonly_unpivoted["task_name"])
    # VERY IMPORTANT: the records count column must be the first!
    col = data_count_per_initial_cl.pop(unique_compound_ID_column_name)
    data_count_per_initial_cl =  pd.concat([col, data_count_per_initial_cl], axis = 1, ignore_index = False)
    weights_for_LP = [weights_for_tasks[tn] for tn in list(data_count_per_initial_cl.columns)]
    # Transpose the array, for use in the LP solver function
    data_count_per_initial_cl_transposed_array = np.array(data_count_per_initial_cl, ndmin = 2).transpose()

    path_to_data_counts_csv = path_to_input_csv.replace(".csv", "_clustered_data_counts.csv")
    print("   Saving data counts in ",path_to_data_counts_csv," ...")
    print("")
    data_count_per_initial_cl.to_csv(path_to_data_counts_csv)
    print("   Done.")
    print("")
    # Calculate the data percentages, for intermediate reporting
    data_percentages_per_property = data_count_per_initial_cl.transform(lambda x: x / x.sum())
    # Not necessary: data_percentages_per_property is already a pandas DataFrame
    #df2 = pd.DataFrame(data_percentages_per_property)
    path_to_data_percentages_csv = path_to_input_csv.replace(".csv", "_clustered_data_percentages.csv")
    print("   Saving data percentages in ",path_to_data_percentages_csv," ...")
    print("")
    # Not necessary: data_percentages_per_property is already a pandas DataFrame
    #df2.to_csv(path_to_data_percentages_csv)
    data_percentages_per_property.to_csv(path_to_data_percentages_csv)
    print("   Done.")
    print("")            

    print("Balancing the data (this may take a long time)...")
    print("")

    print("   Starting the linear program solver...")
    print("")

    # Balance the data
    mapping = balance_data_from_tasks_vs_clusters_array_pulp(
        tasks_vs_clusters_array = data_count_per_initial_cl_transposed_array,
        sizes = sizes,
        task_weights = weights_for_LP,
        equal_weight_perc_compounds_as_tasks = equal_weight_perc_compounds_as_tasks,
        relative_gap = relative_gap,
        time_limit_seconds = time_limit_seconds,
        max_N_threads = max_N_threads)

    if (len(mapping) < data_count_per_initial_cl_transposed_array.shape[1]):
        raise ValueError("The linear program solver did not reach a feasible solution within the specified time limit. \
            Please increase it. \
            To avoid rerunning the whole process, you can use function 'balance_data_from_csv_file_pulp' on \
            saved file "+path_to_data_counts_csv)

    print("Balancing completed.")
    print("")
    
    # Store in df to which final set each initial cluster maps to
    initial_subset_values = list(data_percentages_per_property.index.values)
    final_subset_vs_initial_subset_dict = dict(zip(initial_subset_values, mapping))
    df[ML_subset_column_name] = [final_subset_vs_initial_subset_dict[i] for i in df[initial_cluster_column_name]]
    
    # Calculate the final data summary
    
    print("Calculating final data summary (counts and percentages)...")
    print("")
    
    # First, map the ML subset numbers onto the pivoted data
    df_dataonly_unpivoted[ML_subset_column_name] = [final_subset_vs_initial_subset_dict[i] for i in df_dataonly_unpivoted[initial_cluster_column_name]]

    # Use cross tabulation to make the array of counts of data per cluster per task
    data_count_per_ML_subset = pd.crosstab(index = df_dataonly_unpivoted[ML_subset_column_name], columns = df_dataonly_unpivoted["task_name"])
    # VERY IMPORTANT: the records count column must be the first!
    col = data_count_per_ML_subset.pop(unique_compound_ID_column_name)
    data_count_per_ML_subset =  pd.concat([col, data_count_per_ML_subset], axis = 1, ignore_index = False)
    data_percentages_per_property = data_count_per_ML_subset.transform(lambda x: x / x.sum())
    # Not necessary: data_percentages_per_property is already a pandas DataFrame
    #df2 = pd.DataFrame(data_percentages_per_property)
    
    # Calculate the weighted sum of absolute differences, for reporting
    
    # fractional_sizes = sizes / np.sum(sizes) # already done above
    sk_harmonic = (1 / fractional_sizes) / np.sum(1 / fractional_sizes)
    # debug - probably this is the mistake - ML_subset_column_name does not exist in this summary df2, it's row names
    #absdiffs = np.abs(df2.drop(ML_subset_column_name, axis = 1) - np.array([fractional_sizes]).transpose())
    # Not necessary: data_percentages_per_property is already a pandas DataFrame
    #absdiffs = np.abs(df2 - np.array([fractional_sizes]).transpose())
    absdiffs = np.abs(data_percentages_per_property - np.array([fractional_sizes]).transpose())
    wsabsdiffs = np.dot(np.atleast_2d(sk_harmonic), absdiffs).sum()    
    
    # If required, calculate the min inter-subsets distances

    if (calculate_min_inter_subset_TanimotoDists == True):

        print("Calculating inter-ML-subsets similarities (this may take a long time)...")
        # print("")        
        
        minisdists = min_intercluster_distances(list(df[ML_subset_column_name]), fps)
        df[min_is_dist_column_name] = minisdists
        
        # Calculate the median of the minimal inter-subset distances, per ML_subset
        
        median_dist_per_ML_subset = list(df[[min_is_dist_column_name,ML_subset_column_name]].groupby([ML_subset_column_name]).median()[min_is_dist_column_name])
        min_median_dist_per_ML_subset = min(median_dist_per_ML_subset)
    else:
        median_dist_per_ML_subset = []
        min_median_dist_per_ML_subset = 0

    # Do not alter the user's path_to_output_csv; the number of clusters can always be calculated a posteriori
    #repl_text = "_init_N_cl_" + str(Nc) + ".csv"
    #path_to_output_csv_this_Nc = path_to_output_csv.replace(".csv", repl_text)
    #print("Saving main output csv file in ",path_to_output_csv_this_Nc," ...")

    # New in v19: if an ID was defined, hence dictionary ID_vs_index_dict exists, we need to map it back to df before saving
    # otherwise we drop the temporary column made from the row numbers    
    try:
        df[unique_compound_ID_column_name] = df[unique_compound_ID_column_name].apply(lambda x : ID_vs_index_dict[x])
    except:
        df.drop(columns = [unique_compound_ID_column_name], inplace = True)    
    
    print("Saving main output csv file in ",path_to_output_csv," ...")
    print("")
    # The use of sparse DataFrame columns causes horrible slowness in to_csv.
    # To counteract that, the file is written out in chunks, using an appropriately defined function.
    #df.to_csv(path_to_output_csv, index = False)
    write_csv_from_dataframe_with_sparse_cols(
        dataframe = df,
        sparse_columns_names = cols_with_data,
        output_csv_file_full_path = path_to_output_csv)
    
    if interpret_censored_floats_as_floats == True:
        # Uncensor in df all tasks that were found to be fully continuous
        for tn in unique_task_names:        
            if tn != unique_compound_ID_column_name :
                if type_vs_task_name_dict[tn] == 'cont':
                    vals_u = df[tn].apply(lambda v: float_unc(v, True))
                    df[tn] = pd.arrays.SparseArray([u if u != '' else pd.NA for u in vals_u])
        path_to_output_csv_uncensored = path_to_output_csv.replace('.csv', '_uncensored.csv')
        print("Saving main *uncensored* output csv file in ",path_to_output_csv_uncensored," ...")
        print("")
        # The use of sparse DataFrame columns causes horrible slowness in to_csv.
        # To counteract that, the file is written out in chunks, using an appropriately defined function.
        #df.to_csv(path_to_output_csv, index = False)
        write_csv_from_dataframe_with_sparse_cols(
            dataframe = df,
            sparse_columns_names = cols_with_data,
            output_csv_file_full_path = path_to_output_csv_uncensored)

    if (save_data_summaries == True):
        # Do not alter the user's path_to_output_csv; the number of clusters can always be calculated a posteriori
        #repl_text = "_init_N_cl_" + str(Nc) + "_data_counts.csv"
        #path_to_data_counts_csv = path_to_output_csv.replace(".csv", repl_text)
        path_to_data_counts_csv = path_to_output_csv.replace(".csv", "_data_counts.csv")
        print("Saving data counts in ",path_to_data_counts_csv," ...")
        print("")
        data_count_per_ML_subset.to_csv(path_to_data_counts_csv)
        print("Done.")
        print("")
        # Do not alter the user's path_to_output_csv; the number of clusters can always be calculated a posteriori
        #repl_text = "_init_N_cl_" + str(Nc) + "_data_percentages.csv"
        #path_to_data_percentages_csv = path_to_output_csv.replace(".csv", repl_text)
        path_to_data_percentages_csv = path_to_output_csv.replace(".csv", "_data_percentages.csv")
        print("Saving data percentages in ",path_to_data_percentages_csv," ...")
        print("")
        # Not necessary: data_percentages_per_property is already a pandas DataFrame
        #df2.to_csv(path_to_data_percentages_csv)
        data_percentages_per_property.to_csv(path_to_data_percentages_csv)
        print("Done.")
        print("")
        print("All processes completed.")
        
    return([Nc, median_dist_per_ML_subset, min_median_dist_per_ML_subset, wsabsdiffs, xys])

# 1.2. Sphere exclusion clustering function

def sphere_exclusion_clustering(list_of_molecules = None,
                                list_of_fingerprints = None,
                                N_centers_to_pick = None,
                                min_dist = None,
                                assign_to_initial_centers_only = True,
                                similarity_aggregation = 'mean',
                                folded_to_BitVect = True,
                                morgan_radius = 3,
                                morgan_NBits = 32768,
                                seed = -1):
    """
    Takes a list of molecules or fingerprints, and outputs the list of clusters they belong to.
    
    Parameters
    ----------
    list_of_molecules : list of rdkit molecule objects
        a list of molecules as can be generated by rdkit.Chem.MolFromSmiles
    list_of_fingerprints : list of rdkit Fingerprint objects
        a list of fingerprints, as can be generated by various functions in rdkit.Chem.rdMolDescriptors
    - either molecules or fingerprints must be provided; if fingerprints are provided, they will be used, not recalculated
    - NOTE: if fingerprints are provided, they must be of the correct type! (depending on boolean folded_to_BitVect)
    N_centers_to_pick : positive integer
        number of molecules to select initially by MaxMin
    min_dist : float between 0 and 1
        minimal acceptable Tanimoto fingerprint distance between any two molecules in the initial MaxMin selection
    - either N_centers_to_pick or min_dist must be given; if N_centers_to_pick is specified, min_dist will be ignored
    assign_to_initial_centers_only : bool
        must the post-MaxMin assignment use only the initial centres, or all molecules?
        - after the MaxMin centers are found, clusters are formed with each of them as single member;
        - all remaining molecules must then be assigned to one of these clusters;
        - normally (if this parameter is True), the process assigns a molecule to the *initial center* it is closest to
        - alternatively (if this parameter is False), at each step the similarity of the new mol to *all already assigned* mols
          is calculated, a per-cluster statistics is calculated, depending on the setting of parameter 'similarity_aggregation';
          the new mol is then assigned to the cluster of the molecule to which it has the maximal aggregated statistics.
        - using only the initial centers will give a much faster assignment, but molecules in different clusters might be close;
          this is because 2 mols A and B can be very close between them, even when A is closer to center X and B to center Y.
    similarity_aggregation : string {'max', 'mean', 'median'}
        - only applicable when assign_to_initial_centers_only == False
        - determines what type of aggregation is used to calculate the similarity of a new mol to assign to the set of already
          assigned mols
        - 'max' is the fastest option, no aggregation is needed: the new mol is assigned to the cluster of the mol it is closest to
        - 'mean' or 'median' are slower, as at each step the mean or median of the similarities of the new mol to the already
          assigned mols, grouped by the assigned mols' cluster, must be calculated; but the results might be better.
    folded_to_BitVect : bool
        must the fingerprints be calculated as folded bit vectors?
        - if True, the calculations will be faster but the results less accurate
        - if False, full fingerprints will be used, yielding more accurate results
    morgan_radius : integer {1, 2, 3}
        the radius for the morgan fingerprint calculation, where applicable
    morgan_NBits : positive integer
        the number of bits for the folded fingerprints (only applicable when folded_to_BitVect == True)
    seed : integer
        indicating the seed for the stochastic selection process
        - if == -1, the process will pick a different starting point each time, producing different
          clusters at each run of the same data; if a fixed number, the clusters will be the same at each run.

    Returns
    -------
    List of integer cluster indices each input molecule or fingerprint belongs to (in the same order).
    """
    # START PROCESSING
    
    # Check that the similarity aggregation method is correctly defined, when applicable
    
    if ((assign_to_initial_centers_only == False) & (similarity_aggregation not in {'max','mean','median'})):
        raise ValueError("The similarity aggregation parameter can only be 'max', 'mean' or 'median'.")
    
    # Get or calculate the fingerprints
    
    if (list_of_fingerprints is None):
        if (list_of_molecules is None):        
            raise ValueError("Either fingerprints or molecules must be provided.")
        else:
            if (folded_to_BitVect == True):
                # print("Calculating folded fingerprints...")
                mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius = morgan_radius, fpSize = morgan_NBits)
                # New in v20.2
                #fps = [rdMolDescriptors.GetMorganFingerprintAsBitVect(m, morgan_radius, morgan_NBits) for m in list_of_molecules]
                fps = [mfpgen.GetFingerprint(m) for m in list_of_molecules]
            else:
                # print("Calculating whole fingerprints...")
                mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius = morgan_radius)
                # New in v20.2
                #fps = [Chem.rdMolDescriptors.GetMorganFingerprint(m, morgan_radius, useCounts = False) for m in list_of_molecules]
                fps = [mfpgen.GetSparseFingerprint(m) for m in list_of_molecules]
    else:
        fps = list_of_fingerprints
        
    N0 = len(fps)
    
    # Determine which picker to use, depending on the user setting and type of fingerprint
    
    if (N_centers_to_pick is None):
        
        if (min_dist is None):            
            raise ValueError("Either the number of centers to pick or the minimal distance must be given.")
        else:
            
            if (min_dist < 0) | (min_dist > 1):
                raise ValueError("min_dist is a Tanimoto distance, it cannot be outside [0,1].")
            
            case = 'A'
    else:       
        
        if (N_centers_to_pick < 2) | (N_centers_to_pick >= N0):
            raise ValueError("The number of centers to pick cannot be less than 2 or larger than the total number of molecules in the set.")
        
        case = 'C'
    
    if (folded_to_BitVect == False):
        
        def distfun(fp_index1,fp_index2):
            d = 1 - DataStructs.TanimotoSimilarity(fps[fp_index1],fps[fp_index2])
            return d
        
        if (case == 'A'):
            case = 'B'
        else:
            case = 'D'

    # print("case = ",case)
    print("      Initial MaxMin selection...")
    #print("")

    # Make the initial MaxMin selection
    
    if (case == 'A'):
        picks = rdSimDivPickers.LeaderPicker().LazyBitVectorPick(fps, poolSize = N0, threshold = min_dist)
    elif (case == 'B'):
        picks = rdSimDivPickers.LeaderPicker().LazyPick(distFunc = distfun, poolSize = N0, threshold = min_dist)
    elif (case == 'C'):
        picks = rdSimDivPickers.MaxMinPicker().LazyBitVectorPick(fps, poolSize = N0, pickSize = N_centers_to_pick, seed = seed)
    elif (case == 'D'):
        picks = rdSimDivPickers.MaxMinPicker().LazyPick(distFunc = distfun, poolSize = N0, pickSize = N_centers_to_pick, seed = seed)
    
    N = len(picks)
    
    print("      # items selected by MaxMin = ", N)
    #print("")

    if (N <= 1):
        raise ValueError("No molecules could be selected. Please review your input parameters.")
    
    # Make the lists of assigned molecules, their fingerprints and cluster numbers
    
    assigned = list(picks)
    clusters = [i for i in range(N)]
    cfps = [fps[i] for i in assigned]
    
    # Make the list of unassigned molecules
    
    unassigned = [i for i in range(N0) if i not in assigned]

    print("      Assignment of remaining items...")
    #print("")

    # Assign the unassigned molecules to the initial clusters
    
    for i in unassigned:
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], cfps)
        if ((assign_to_initial_centers_only == True) | (similarity_aggregation == 'max')):
            closest_pick_index = sims.index(max(sims))
        else:
            closest_pick_index = pd.Series(sims).groupby(clusters).agg(similarity_aggregation).sort_values(ascending=False).index[0]
        clusters.append(clusters[closest_pick_index])
        assigned.append(i)
        if (assign_to_initial_centers_only == False):
            cfps.append(fps[i])
    
    # Create the mapping of the input records to final clusters, and output it
    
    mapping = [cl for _, cl in sorted(zip(assigned, clusters))]
    
    return mapping

# 1.2.bis MaxMin selection function - in case one only wants cluster centres

def MaxMin_selection(
    list_of_molecules = None,
    list_of_fingerprints = None,
    N_centers_to_pick = None,
    min_dist = None,
    folded_to_BitVect = True,
    morgan_radius = 3,
    morgan_NBits = 32768,
    seed = -1):
    """
    Takes a list of molecules or fingerprints, and outputs the list of indices of the selected
    ones, according to MaxMin (essentially, a diverse selection).
    
    Parameters
    ----------
    list_of_molecules : list of rdkit molecule objects
        a list of molecules as can be generated by rdkit.Chem.MolFromSmiles
    list_of_fingerprints : list of rdkit Fingerprint objects
        a list of fingerprints, as can be generated by various functions in rdkit.Chem.rdMolDescriptors
    - either molecules or fingerprints must be provided; if fingerprints are provided, they will be used, not recalculated
    - NOTE: if fingerprints are provided, they must be of the correct type! (depending on boolean folded_to_BitVect)
    N_centers_to_pick : positive integer
        number of molecules to select initially by MaxMin
    min_dist : float between 0 and 1
        minimal acceptable Tanimoto fingerprint distance between any two molecules in the initial MaxMin selection
    - either N_centers_to_pick or min_dist must be given; if N_centers_to_pick is specified, min_dist will be ignored
    folded_to_BitVect : bool
        must the fingerprints be calculated as folded bit vectors?
        - if True, the calculations will be faster but the results less accurate
        - if False, full fingerprints will be used, yielding more accurate results
    morgan_radius : integer {1, 2, 3}
        the radius for the morgan fingerprint calculation, where applicable
    morgan_NBits : positive integer
        the number of bits for the folded fingerprints (only applicable when folded_to_BitVect == True)
    seed : integer
        indicating the seed for the stochastic selection process
        - if == -1, the process will pick a different starting point each time, producing different
          clusters at each run of the same data; if a fixed number, the clusters will be the same at each run.

    Returns
    -------
    List of indices of the selected input molecules/fingerprints (in the same order).
    """    
    # START PROCESSING

    # Get or calculate the fingerprints
    
    if (list_of_fingerprints is None):
        if (list_of_molecules is None):        
            raise ValueError("Either fingerprints or molecules must be provided.")
        else:
            if (folded_to_BitVect == True):
                # print("Calculating folded fingerprints...")
                mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius = morgan_radius, fpSize = morgan_NBits)
                # New in v20.2
                #fps = [rdMolDescriptors.GetMorganFingerprintAsBitVect(m, morgan_radius, morgan_NBits) for m in list_of_molecules]
                fps = [mfpgen.GetFingerprint(m) for m in list_of_molecules]
            else:
                # print("Calculating whole fingerprints...")
                mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius = morgan_radius)
                # New in v20.2
                #fps = [Chem.rdMolDescriptors.GetMorganFingerprint(m, morgan_radius, useCounts = False) for m in list_of_molecules]     
                fps = [mfpgen.GetSparseFingerprint(m) for m in list_of_molecules]
    else:
        fps = list_of_fingerprints
        
    N0 = len(fps)
    
    # Determine which picker to use, depending on the user setting and type of fingerprint
    
    if (N_centers_to_pick is None):
        
        if (min_dist is None):            
            raise ValueError("Either the number of centers to pick or the minimal distance must be given.")
        else:
            
            if (min_dist < 0) | (min_dist > 1):
                raise ValueError("min_dist is a Tanimoto distance, it cannot be outside [0,1].")
            
            case = 'A'
    else:       
        
        if (N_centers_to_pick < 2) | (N_centers_to_pick >= N0):
            raise ValueError("The number of centers to pick cannot be less than 2 or larger than the total number of molecules in the set.")
        
        case = 'C'
    
    if (folded_to_BitVect == False):
        
        def distfun(fp_index1,fp_index2):
            d = 1 - DataStructs.TanimotoSimilarity(fps[fp_index1],fps[fp_index2])
            return d
        
        if (case == 'A'):
            case = 'B'
        else:
            case = 'D'

    # print("case = ",case)
    
    # Make the initial MaxMin selection
    
    if (case == 'A'):
        picks = rdSimDivPickers.LeaderPicker().LazyBitVectorPick(fps, poolSize = N0, threshold = min_dist)
    elif (case == 'B'):
        picks = rdSimDivPickers.LeaderPicker().LazyPick(distFunc = distfun, poolSize = N0, threshold = min_dist)
    elif (case == 'C'):
        picks = rdSimDivPickers.MaxMinPicker().LazyBitVectorPick(fps, poolSize = N0, pickSize = N_centers_to_pick, seed = seed)
    elif (case == 'D'):
        picks = rdSimDivPickers.MaxMinPicker().LazyPick(distFunc = distfun, poolSize = N0, pickSize = N_centers_to_pick, seed = seed)
    
    N = len(picks)
    
    if (N <= 1):
        raise ValueError("No molecules could be selected. Please review your input parameters (perhaps the min_dist is too high?).")
  
    return list(picks)

# 1.3. Iterative minimal overlap clustering function

def iterative_clustering_by_minimal_overlap(list_of_molecules = None,
                                       list_of_fingerprints = None,
                                       priority_to_removal_of_bits_from_S0 = False,
                                       min_overlap = 5,
                                       min_sim = 0.56,
                                       folded_to_BitVect = True,
                                       morgan_radius = 3,
                                       morgan_NBits = 32768):
    """
    Takes a list of molecules or fingerprints, and clusters them according to chemistry, by a 'minimal overlap' method.
    
    Parameters
    ----------
    list_of_molecules : list of rdkit molecule objects
        a list of molecules as can be generated by rdkit.Chem.MolFromSmiles
    list_of_fingerprints : list of rdkit Fingerprint objects
        a list of fingerprints, as can be generated by various functions in rdkit.Chem.rdMolDescriptors
    - either molecules or fingerprints must be provided; if fingerprints are provided, they will be used, not recalculated
    - NOTE: if fingerprints are provided, they must be of the correct type! (depending on boolean folded_to_BitVect)
    priority_to_removal_of_bits_from_S0 : bool
        - if True, when choosing different molecules result in the same change in overlap,
          preference will be given to molecules that remove as many bits as possible from the starting set
        - if False, preference will be given to molecules that add as few bits as possible to the new set
    min_overlap : integer >= 0
        - for a new potential molecule to be added to an existing cluster, the number of bits that
          it has in common with the cluster is calculated --> matches;
        - min_overlap is the minimal value of matches for which a molecule is added to the cluster,
          otherwise it starts its own new cluster.
        - a smaller min_overlap will make fewer, larger clusters with fewer bits in common to all mols;
        - a larger min_overlap will make more, smaller clusters with more bits in common to all mols;
        - an excessively large min_overlap will make ~1 cluster per molecule.
        NOTE: if you want min_sim to control the process, set min_overlap to 0.
    min_sim : float between 0 and 1
        - ratio of matches (defined above) to the number of bits that are common to all molecules
          in the current cluster --> sim.
        - if the current cluster has only 1 molecule, this parameter is ignored.
        - otherwise, the compound is added to the cluster only if it meets the min_overlap requirement
          AND its sim is >= min_sim
        - this parameter essentially wants to prevent that a cluster has many molecules with many common
          bits, and a new compound comes in, which has only very few of those bits in common with it,
          and redefines the cluster in too narrow a way.
        - a smaller min_sim will make fewer, larger clusters with fewer bits in common to all mols;
        - a larger min_sim will make more, smaller clusters with more bits in common to all mols;
        - an excessively large min_sim (close to 1) will tend to make ~1 cluster per molecule.
        NOTE: if you want min_overlap to control the process, set min_sim to 0.
    folded_to_BitVect : bool
        must the fingerprints be calculated as folded bit vectors?
        - if True, the calculations will be faster but the results less accurate
        - if False, full fingerprints will be used, yielding more accurate results
    morgan_radius : integer {1, 2, 3}
        the radius for the morgan fingerprint calculation, where applicable
    morgan_NBits : positive integer
        the number of bits for the folded fingerprints (only applicable when folded_to_BitVect == True)
    seed : integer
        indicating the seed for the stochastic selection process
        - if == -1, the process will pick a different starting point each time, producing different
          clusters at each run of the same data; if a fixed number, the clusters will be the same at each run.
    
    Returns
    -------
    List of lists:
    0) the subset they belong to
    1) the order of selection
    2) the current overlap
    3) the current number of bits in S0
    4) the current number of bits in S1
    5) the current size of S0
    6) the current size of S1
    7) the current Shannon entropy of S0
    8) the current Shannon entropy of S1
    9) the current Shannon entropy change of S0
    10) the current Shannon entropy change of S1
    """
    # START PROCESSING
    
    # Get or calculate the fingerprints
    
    if (list_of_fingerprints is None):
        if (list_of_molecules is None):        
            raise ValueError("Either fingerprints or molecules must be provided.")
        else:
            if (folded_to_BitVect == True):
                # print("Calculating folded fingerprints...")
                mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius = morgan_radius, fpSize = morgan_NBits)
                # New in v20.2
                #fps = [rdMolDescriptors.GetMorganFingerprintAsBitVect(m, morgan_radius, morgan_NBits) for m in list_of_molecules]
                fps = [mfpgen.GetFingerprint(m) for m in list_of_molecules]
            else:
                # print("Calculating whole fingerprints...")
                mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius = morgan_radius)
                # New in v20.2
                #fps = [Chem.rdMolDescriptors.GetMorganFingerprint(m, morgan_radius, useCounts = False) for m in list_of_molecules]
                fps = [mfpgen.GetSparseFingerprint(m) for m in list_of_molecules] 
    else:
        fps = list_of_fingerprints
        #del(list_of_fingerprints)
    
    # Identify the type of fingerprint
    
    allowed_bit_vector_classes = ["<class 'rdkit.DataStructs.cDataStructs.ExplicitBitVect'>",
                                  "<class 'rdkit.DataStructs.cDataStructs.SparseBitVect'>"]
    
    allowed_integer_vector_classes = ["<class 'rdkit.DataStructs.cDataStructs.UIntSparseIntVect'>",
                                     "<class 'rdkit.DataStructs.cDataStructs.ULongSparseIntVect'>",
                                     "<class 'rdkit.DataStructs.cDataStructs.LongSparseIntVect'>",
                                     "<class 'rdkit.DataStructs.cDataStructs.IntSparseIntVect'>"]
    
    if (str(type(fps[0])) in allowed_bit_vector_classes):
        case = "BitVect"
    elif (str(type(fps[0])) in allowed_integer_vector_classes):
        case = "IntVect"
    else:
        raise ValueError("The fingerprints are not any of the allowed bit or integer vector types.")
    
    # 4. Make fingerprint summaries as needed for the selection.

    # 4.1. Create a dictionary with the global counts fingerprint 'fp0' for the initial set S0 with all molecules.
    # At the same time create a dictionary 'id_vs_fpbit' of which molecules contain which fpbits.
    # Also, make a dictionary of the list of bits of each compound, for easier reference.
    
    fp0 = dict()
    id_vs_fpbit = dict()
    fp_vs_id = dict()
    
    for i, fpi in enumerate(fps):
        
        if (case == "BitVect"):
            fpi_bits_list = [fpi.GetOnBits()[j] for j in range(fpi.GetNumOnBits())]
        else:
            fpi_bits_list = list(fpi.GetNonzeroElements().keys())            
        
        fp_vs_id[i] = fpi_bits_list
        
        for fpbit in fpi_bits_list:
            if fpbit in fp0:
                fp0[fpbit] += 1
            else:
                fp0[fpbit] = 1
            if fpbit in id_vs_fpbit:
                id_vs_fpbit[fpbit].append(i)
            else:
                id_vs_fpbit[fpbit] = [i]
    
    # 4.1.2. Make a frequencies dictionary, to speed up the calculation of Shannon entropy
    
    fp0_freqs = dict()
    
    for f in list(fp0.values()):
        if f in fp0_freqs:
            fp0_freqs[f] += 1
        else:
            fp0_freqs[f] = 1
    
    # Function definition no longer necessary thanks to the new SE calculation method
    #def Fr0(f):
    #    if f in fp0_freqs:
    #        Fr = fp0_freqs[f]
    #    else:
    #        Fr = 0
    #    return Fr
    #def Fr1(f):
    #    if f in fp1_freqs:
    #        Fr = fp1_freqs[f]
    #    else:
    #        Fr = 0
    #    return Fr
    
    # 4.1.3. Make a U(f) list for all required values of f (from 2 to len(fps)), so it's done only once
    # U(f) = f*log(f)
    
    Uf = [0] * (len(fps)+1)
    
    for f in range(2,len(fps)+1):
        Uf[f] = f * np.log2(f)
    
    # 4.2. Create a dictionary 'fp01' of fingerprint bits with value 1 in fp0 (i.e. those that would disappear if a compound
    # containing them were moved from S0 to S1).

    fp01 = {fpbit for fpbit,value in fp0.items() if value == 1}
    fp01 = dict.fromkeys(fp01, 0)
    
    # 4.3. For each compound, calculate how many bits it would remove from S0 if it were moved to S1, and store its negated value.
    # And calculate how many new bits it would add to S1 if it were moved to S1.
    # At this stage, S1 is empty, so this is simply the total number of bits in the compound.

    deltaS0 = []
    deltaS1 = []
    for i in fp_vs_id:
        deltaS0.append(0)
        for fpbit in fp_vs_id[i]:
            if fpbit in fp01:
                deltaS0[i] -= 1
        deltaS1.append(len(fp_vs_id[i]))
        
    # The delta overlap is the element-wise sum of the elements of deltaS0 and deltaS1

    delta_overlap = []
    for (dS0, dS1) in zip(deltaS0, deltaS1):
        delta_overlap.append(dS0+dS1)

    # 5. Make the partitions
    
    # For a single bipartition:

    # Initialise a dictionary S1, which will contain the indices of the compounds in S1, and a fingerprint fp1, for the bits in S1.
    # At each step, identify which compound corresponds to the minimal delta_overlap, resolving ties as specified by the user, and:
    # - using the fingerprint bits in the moved compound, update fp1, fp0, fp01, deltaS0, deltaS1 and delta_overlap
    # - loop, until a condition is met
    # - append the compound index to S1
    # - make sure that the delta_overlap of the selected item is set to a very high value, so it's not chosen

    # Once a bipartition is complete, reinitialise all the required dictionaries and start again.

    Nk = len(fps)
    
    # 5.2. Run the iterations

    clusters = defaultdict(list)
    order_of_selection_list = []
    current_overlap_list = []
    current_N_bits_S0_list = []
    current_N_bits_S1_list = []
    current_size_S0_list = []
    current_size_S1_list = []
    current_ShannonEntropy_S0_list = []
    current_ShannonEntropy_S1_list = []
    current_ShannonEntropy_S0_delta_list = []
    current_ShannonEntropy_S1_delta_list = []
    set_n = 0
    already_assigned = dict()
    order_of_sel = 0
    Nk_S0 = Nk
    absolute_sizes = []
        
    while (order_of_sel < Nk):

        set_n += 1
        S1 = dict()
        Nk_S1 = 0
        fp1 = dict()
        fp1_freqs = dict()
        total_overlap = [0]
        selection_iteration = 0
        num_bits_in_S0 = len(fp0)
        num_bits_in_S1 = len(fp1)
        num_bits_total = num_bits_in_S0
        current_overlap = num_bits_in_S0 + num_bits_in_S1 - num_bits_total
        # Calculation of of SE from bit frequencies (slow, only for initial testing)
        #SE_S0_from_bits = -1 * (sum([(fp0[b]/Nk_S0)*np.log2(fp0[b]/Nk_S0) for b in fp0 if fp0[b] > 0]) + \
        #              sum([(1-fp0[b]/Nk_S0)*np.log2(1-fp0[b]/Nk_S0) for b in fp0 if fp0[b] < Nk_S0]))
        SE_S1 = 0
        # SE calculated by the frequencies list method (faster)
        if (Nk_S0 > 0):
        #    SE_S0 = num_bits_in_S0 * np.log2(Nk_S0) - \
        #    (Fr0(Nk_S0) * Uf[Nk_S0] + 
        #     sum((Fr0(f)+Fr0(Nk_S0-f))*Uf[f] for f in set(list(fp0_freqs.keys())+[Nk_S0-f for f in fp0_freqs.keys()]) if ((f >= 2) & (f <= Nk_S0-1))) ) \
        #    / Nk_S0
        # New version, does not require the complicated set detection or Fr definition
            SE_S0 = num_bits_in_S0 * np.log2(Nk_S0) -             sum((Uf[f]+Uf[Nk_S0-f])*Fr for f,Fr in fp0_freqs.items())             / Nk_S0
        else:
            SE_S0 = 0
        
        #print("Starting cluster # ",set_n,"...")
        #print("Initial situation:")
        #print("num_bits_in_S0 = ", num_bits_in_S0)
        #print("num_bits_in_S1 = ", num_bits_in_S1)
        #print("current overlap = ", current_overlap)
        #print("SE_S0 from freqs = ", SE_S0)
        #print("SE_S0 from bits = ", SE_S0_from_bits)
        
        while (order_of_sel < Nk):
            
            # The compound to move from S0 to S1 must have the lowest delta_overlap

            # sel = delta_overlap.index(min(delta_overlap))
            # Note: no longer based on delta_overlap alone; now ties are broken by deltaS, see below

            # Identify first the indices of all compounds with minimal delta_overlap
            min_dov = min(delta_overlap)
            min_dov_indices = [i for i, dov in enumerate(delta_overlap) if dov == min_dov]
            # Then find which has the lowest deltaS0 (if priority_to_removal_of_bits_from_S0 == True)
            # or the lowest deltaS1 (if priority_to_removal_of_bits_from_S0 == False)
            if (priority_to_removal_of_bits_from_S0 == True):
                deltaS = [deltaS0[i] for i in min_dov_indices]
            else:
                deltaS = [deltaS1[i] for i in min_dov_indices]
            # Select the compound to move
            sel = min_dov_indices[deltaS.index(min(deltaS))]
            
            # Check if the new compound (sel) must be included in S1 or start a new cluster of its own
            include = 0
            # When S1 is empty, sel must be added to S1 by necessity
            # At this point, also create the first current set of bits in S1 (the bits in sel)
            if (Nk_S1 == 0):
                include = 1
                bits_in_common = fp_vs_id[sel]
                acceptable_bits_for_S1 = dict.fromkeys(bits_in_common,1)
            else:
                # when S1 has at least 1 compound, find the bit overlap with sel ('matches')
                # and the ratio of matches to the current # bits in common to all S1 compounds ('sim')
                bits_in_common = [fpbit for fpbit in fp_vs_id[sel] if fpbit in acceptable_bits_for_S1]
                matches = len(bits_in_common)
                sim = matches / len(acceptable_bits_for_S1)
                
                # when S1 has only 1 compound, sel is added to S1 if it has enough bits in common
                if ((Nk_S1 == 1) & (matches >= min_overlap)):
                    include = 1
                else:
                    # when S1 has 2 compounds or more, sel is added to S1 if it has enough bits
                    # in common AND if its common bits are a sufficient proportion of the current ones
                    if ((matches >= min_overlap) & (sim >= min_sim)):
                        include = 1
                        
            # if include == 0 exit this inner loop to store the results and reinitialise S1
            if (include == 0):
                #print("Closing cluster ",set_n," (",Nk_S1,") compounds.")
                #print("Remaining compounds = ",(Nk-order_of_sel))
                break
            else:
                # otherwise, update the bits that must be present in the next compound to go into S1
                acceptable_bits_for_S1 = dict.fromkeys(bits_in_common,1)
            
            # and add the selected compound to S1

            order_of_sel += 1
            order_of_selection_list.append(order_of_sel)
            
            # Process the fingerprint of the selected compound
            current_overlap_from_delta = total_overlap[selection_iteration]+delta_overlap[sel]
            total_overlap.append(current_overlap_from_delta)

            selection_iteration += 1
            Nk_S0 -= 1
            Nk_S1 += 1
            
            for fpbit in fp_vs_id[sel]:

                # Add the bits in the selected compound to fp1
                if fpbit in fp1:
                    fp1[fpbit] += 1
                else:
                    fp1[fpbit] = 1
                    num_bits_in_S1 += 1
                    # If the bit is new to fp1, the deltaS1 of all (still selectable) compounds containing this bit must decrease by 1
                    for x in id_vs_fpbit[fpbit]:
                        if x not in already_assigned:
                            deltaS1[x] -= 1
                            delta_overlap[x] -= 1
                
                # Adding this bit to fp1 increases by 1 the count of its new (current) frequency
                if fp1[fpbit] in fp1_freqs:
                    fp1_freqs[fp1[fpbit]] += 1
                else:
                    fp1_freqs[fp1[fpbit]] = 1
                # And it decreases by 1 the count of its previous frequency, except if it is 0
                if fp1[fpbit] > 1:
                    fp1_freqs[fp1[fpbit]-1] -= 1
                    if fp1_freqs[fp1[fpbit]-1] == 0:
                        fp1_freqs.pop(fp1[fpbit]-1)
                
                # Remove the bits in the selected compound from fp0
                fp0[fpbit] -= 1
                if (fp0[fpbit] == 1):
                    fp01[fpbit] = 0
                    # If the bit reaches count 1 in fp0, the deltaS0 of all (still selectable) compounds containing this bit must decrease by 1
                    for x in id_vs_fpbit[fpbit]:
                        if x not in already_assigned:
                            deltaS0[x] -= 1
                            delta_overlap[x] -= 1
                elif (fp0[fpbit] == 0):
                    fp01.pop(fpbit)
                    num_bits_in_S0 -= 1
                    # If the bit reaches count 0 in fp0, the deltaS0 of all (still selectable) compounds containing this bit must increase by 1
                    for x in id_vs_fpbit[fpbit]:
                        if x not in already_assigned:
                            deltaS0[x] += 1
                            delta_overlap[x] += 1
                
                # Removing this bit from fp0 decreases by 1 the count of its previous frequency
                fp0_freqs[fp0[fpbit]+1] -= 1
                if fp0_freqs[fp0[fpbit]+1] == 0:
                    fp0_freqs.pop(fp0[fpbit]+1)
                # And it increases by 1 the count of its new (current) frequency, except if it is 0
                if fp0[fpbit] > 0:
                    if fp0[fpbit] in fp0_freqs:
                        fp0_freqs[fp0[fpbit]] += 1
                    else:
                        fp0_freqs[fp0[fpbit]] = 1

                # Note: in previous implementations, the bits that reached frequency 0 in fp0 were not removed (.pop).
                #       However, not doing that forced one to use conditions like fp0[b] > 0 to avoid errors.
                #       It seems better to remove the bits from fp0. This also ensures len(fp0) = # bits in S0.
                if (fp0[fpbit] == 0):
                    fp0.pop(fpbit)
                                        
            current_overlap = num_bits_in_S0 + num_bits_in_S1 - num_bits_total
            
            # Calculate the current Shannon entropies            
            old_SE_S0 = SE_S0
            old_SE_S1 = SE_S1
            # Calculation of of SE from bit frequencies (slow, only for initial testing)
            #SE_S0 = -1 * (sum([(fp0[b]/Nk_S0)*np.log2(fp0[b]/Nk_S0) for b in fp0 if fp0[b] > 0]) + \
            #              sum([(1-fp0[b]/Nk_S0)*np.log2(1-fp0[b]/Nk_S0) for b in fp0 if fp0[b] < Nk_S0]))
            #SE_S1 = -1 * (sum([(fp1[b]/Nk_S1)*np.log2(fp1[b]/Nk_S1) for b in fp1 if fp1[b] > 0]) + \
            #              sum([(1-fp1[b]/Nk_S1)*np.log2(1-fp1[b]/Nk_S1) for b in fp1 if fp1[b] < Nk_S1]))
            # SE calculated by the frequencies list method (faster)
            if (Nk_S0 > 0):
            #    SE_S0 = num_bits_in_S0 * np.log2(Nk_S0) - \
            #    (Fr0(Nk_S0) * Uf[Nk_S0] + 
            #     sum((Fr0(f)+Fr0(Nk_S0-f))*Uf[f] for f in set(list(fp0_freqs.keys())+[Nk_S0-f for f in fp0_freqs.keys()]) if ((f >= 2) & (f <= Nk_S0-1))) ) \
            #    / Nk_S0
                # New version, does not require the complicated set detection or Fr definition
                SE_S0 = num_bits_in_S0 * np.log2(Nk_S0) -                 sum((Uf[f]+Uf[Nk_S0-f])*Fr for f,Fr in fp0_freqs.items())                 / Nk_S0
            else:
                SE_S0 = 0
            
            #SE_S1 = num_bits_in_S1 * np.log2(Nk_S1) - \
            #(Fr1(Nk_S1) * Uf[Nk_S1] + 
            # sum((Fr1(f)+Fr1(Nk_S1-f))*Uf[f] for f in set(list(fp1_freqs.keys())+[Nk_S1-f for f in fp1_freqs.keys()]) if ((f >= 2) & (f <= Nk_S1-1))) ) \
            #/ Nk_S1
            # New version, does not require the complicated set detection or Fr definition
            SE_S1 = num_bits_in_S1 * np.log2(Nk_S1) -             sum((Uf[f]+Uf[Nk_S1-f])*Fr for f,Fr in fp1_freqs.items())             / Nk_S1
                    
            current_ShannonEntropy_S0_list.append(SE_S0)
            current_ShannonEntropy_S1_list.append(SE_S1)
            delta_SE_S0 = (SE_S0 - old_SE_S0)
            delta_SE_S1 = (SE_S1 - old_SE_S1)
            current_ShannonEntropy_S0_delta_list.append(delta_SE_S0)
            current_ShannonEntropy_S1_delta_list.append(delta_SE_S1)            
            
            # Update the metrics lists
            current_overlap_list.append(current_overlap)
            current_N_bits_S0_list.append(num_bits_in_S0)
            current_N_bits_S1_list.append(num_bits_in_S1)
            current_size_S0_list.append(Nk_S0)
            current_size_S1_list.append(Nk_S1)
                        
            #print("Selected ",selection_iteration,"/",size)
            #print("num_bits_in_S0 = ", num_bits_in_S0)
            #print("num_bits_in_S1 = ", num_bits_in_S1)
            #print("current overlap = ", current_overlap)
            #print("current overlap from delta = ", current_overlap_from_delta)
            #print("SE_S0 from bit fractions = ", SE_S0)
            #print("SE_S0 from freqs = ", SE_S0_from_freqs)
            #print("SE_S1 from bit fractions = ", SE_S1)
            #print("SE_S1 from freqs = ", SE_S1_from_freqs)

            # Append the selected compound to S1 and to already_assigned
            S1[sel] = 1
            already_assigned[sel] = set_n
            # Set the delta_overlap of sel to a very high value
            delta_overlap[sel] = len(fp0)
            
        #print("Done. Final situation:")
        #print("num_bits_in_S0 = ", num_bits_in_S0)
        #print("num_bits_in_S1 = ", num_bits_in_S1)
        #print("current overlap = ", current_overlap)
        # plt.plot(total_overlap); plt.xlabel('iteration'); plt.ylabel('overlap')
        #print("")

        # Append the result of this selection to the main clusters dictionary
        for i in S1:
            clusters[set_n].append(i)
        
        absolute_sizes.append(Nk_S1)
        
        # Reinitialise, of course only if there is another selection to make
        if (order_of_sel < Nk):
            fp00 = [fpbit for fpbit,value in fp0.items() if value == 0]
            for fpbit in fp00:
                fp0.pop(fpbit)
                # id_vs_fpbit.pop(fpbit) # not really necessary; the bits that are no longer in fp0 will not be searched
            # All compounds that are still selectable and whose bits will disappear from S1 when S1 is emptied need to have their
            # deltaS1 and delta_overlap corrected as appropriate.    
            for fpbit in fp1:
                for x in id_vs_fpbit[fpbit]:
                    if x not in already_assigned:
                        deltaS1[x] += 1
                        delta_overlap[x] += 1
            # After which, fp1 and S1 can be safely reinitialised (which happens at the beginning of the loop).
        else:
            #print("Closing cluster ",set_n," (",Nk_S1,") compounds.")
            print("Clustering complete.")
            print("Total # clusters made = ", set_n)

    # END: output
    
    # Create the permutation order mapping the input set to the order of selection
    indices = []
    for c in clusters:
        indices.extend(clusters[c])
    
    order = [x for _,x in sorted(zip(indices, range(Nk)))]
    
    # Map the permutation on all relevant lists
    
    S = len(absolute_sizes)
    cls = np.repeat([i+1 for i in range(S)], absolute_sizes)
    cls = [cls[i] for i in order]
    order_of_selection_list = [order_of_selection_list[i] for i in order]
    current_overlap_list = [current_overlap_list[i] for i in order]
    current_N_bits_S0_list = [current_N_bits_S0_list[i] for i in order]
    current_N_bits_S1_list = [current_N_bits_S1_list[i] for i in order]
    current_size_S0_list = [current_size_S0_list[i] for i in order]
    current_size_S1_list = [current_size_S1_list[i] for i in order]
    current_ShannonEntropy_S0_list = [current_ShannonEntropy_S0_list[i] for i in order]
    current_ShannonEntropy_S1_list = [current_ShannonEntropy_S1_list[i] for i in order]
    current_ShannonEntropy_S0_delta_list = [current_ShannonEntropy_S0_delta_list[i] for i in order]
    current_ShannonEntropy_S1_delta_list = [current_ShannonEntropy_S1_delta_list[i] for i in order]

    # Return the results
    
    return [cls,
            order_of_selection_list,
            current_overlap_list,
            current_N_bits_S0_list,
            current_N_bits_S1_list,
            current_size_S0_list,
            current_size_S1_list,
            current_ShannonEntropy_S0_list,
            current_ShannonEntropy_S1_list,
            current_ShannonEntropy_S0_delta_list,
            current_ShannonEntropy_S1_delta_list]

# 1.4. Linear programming function needed to balance the data while merging clusters

def balance_data_from_tasks_vs_clusters_array_pulp(tasks_vs_clusters_array,
                                              sizes, # list of numbers
                                              task_weights = None, # None or list of numbers
                                              equal_weight_perc_compounds_as_tasks = True,
                                              relative_gap = 0,
                                              time_limit_seconds = 60*60*24*365,
                                              max_N_threads = 4):
    """
    Takes a numpy array with the summary of number of items (data points) per group (cluster) vs property (task),
    and merges the groups (clusters) to a new (smaller) number of groups, by linear programming, aiming to have
    *proportional* distributions of items across final groups, for each property (task).
    
    Example
    -------
           0   1   2   3   4
    'recs' 7   4   7   4   5
    'A'    4   0   2   3   1
    'B'    1   1   5   2   4
    'C'    6   4   3   2   0
    
    This array represents a clustered dataset where:
    - cluster 0 has 7 records, 4 of which have data for property 'A', 1 for 'B' and 6 for 'C'
    - cluster 1 has 4 records, 0 of which have data for property 'A', 1 for 'B' and 4 for 'C'
    - ...
    
    Turning this into fractions (by dividing each row by its *row sum*):
           0      1      2      3      4
    'recs' 0.26   0.15   0.26   0.15   0.18
    'A'    0.40   0.00   0.20   0.30   0.10
    'B'    0.08   0.08   0.38   0.15   0.31
    'C'    0.40   0.27   0.20   0.13   0.00

    - cluster 0 has 26% of all records, 40% of data for property 'A', 8% of data for 'B' and 40% for 'C'
    - cluster 1 has 15% of all records, 0% of data for property 'A', 8% of data for 'B' and 27% for 'C'
    - ...

    The goal may be to make 3 final subsets of similar sizes (i.e. similar number of records).
    If the size must be similar, the fraction of *records* in each final group should be close to 1/3 = 0.333.
    So one needs to choose which initial clusters to combine into each final group to achieve that.
    At the same time, one also wants the fractions of *data* for each property to be close to 0.333 across
    the final groups.
    In practice, the *columns* of the above matrix must be merged in such a way that the resulting sums of
    fractions are as close as possible to the required fractional sizes of the final groups, down each *column*.
    In this example, the fractional sizes are 0.333 for each final group.
        
    By running this function, this example yields : [3, 1, 2, 1, 3], i.e. assign cluster 0 to final group 3,
    cluster 1 to final group 1, cluster 2 to final group 2...
    
           3   1   2   1   3
           0   1   2   3   4
    'recs' 7   4   7   4   5
    'A'    4   0   2   3   1
    'B'    1   1   5   2   4
    'C'    6   4   3   2   0

    Merging:
           1       2     3
           [1,3]   [2]   [0,4]
    'recs' 8       7     12
    'A'    3       2     5
    'B'    3       5     5
    'C'    6       3     6

    Fractions:
           1       2      3
           [1,3]   [2]    [0,4]
    'recs' 0.30    0.26   0.44
    'A'    0.30    0.20   0.50
    'B'    0.23    0.385  0.385
    'C'    0.40    0.20   0.40
    
    This is as close to 'ideal' balance (here, 0.333 everywhere) as one can get with these data.    
    
    Parameters
    ----------
    tasks_vs_clusters_array : np.array
        the summary of number of data points per cluster, per task
        - columns represent clusters
        - rows represent tasks, except the first row, which represents the number of records (or compounds)
        - optionally, instead of the number of data points, the provided array may contain the *percentages*
          of data points _for the task across all clusters_ (i.e. each *row*, NOT column, may sum to 1).
        IMPORTANT: make sure the array has 2 dimensions, even if only balancing the # of data records,
        so there is only 1 row. This can be achieved by setting ndmin = 2 in the np.array function.
    sizes : list of positive numbers
        list of the desired final sizes (will be normalised to fractions internally)
    task_weights : list of numbers of the same length as the number of *rows* of tasks_vs_clusters_array    
        - these are the weights to apply to the absolute differences in % records/data between the
          linear programming solution and the ideal (defined by sizes)
        - like in the array, element 0 of the list refers to the number of records, the rest to to each task    
        - [weights can still be modulated by option 'equal_weight_perc_compounds_as_tasks' below]
        - [the normalisation is done automatically, so these weights are actually seen as relative proportions]
        - in 'normal' cases this is a list of 1's (which is the same as leaving it set to None)
        - assigning individual different weights means establishing relations between the final weights of the various tasks
        - example: 3 tasks --> 4 rows (first row must always refer to the % records by construction)
          --> [1, 1, 1, 1] gives the same initial weight to the % records as to each of the 3 tasks
          --> [1, 1, 0.5, 0.5] gives the same initial weight to the % records as to the 1st task, 
            and half of that weight to the 2nd and 3rd task.
            Such a scheme could be of use e.g. if there were originally only 2 tasks, but the 2nd task was
            a categorical property that was split into 2 sub-tasks, to balance its 2 label values.
            Then, to avoid over-weighting the 2nd task, its original 1 is split into 0.5 + 0.5.
    equal_weight_perc_compounds_as_tasks : bool
        - this parameter is introduced to allow putting more or less emphasis on matching the % records vs the tasks
        - if True, no change to task_weights is made: make sure to encode in it the relative importance of % records upfront
        Example: [1, 1, 1] --> same importance to the % records as to *each* of the 2 tasks
        - if False, task_weights[1:] is summed, and the task_weights[0] is multiplied by that sum
        Example: [1, 1, 1] --> sum of [1:] = 2 --> final weights = [2, 1, 1] --> matching % records has the same weight as *all* other tasks
    relative_gap : float > 0
        - the relative gap between the absolute optimal objective and the current one at which the solver
          stops and returns a solution. Can be very useful for cases where the exact solution requires
          far too long to be found to be of any practical use.
        - set to 0 to obtain the absolute optimal solution (if reached within the time_limit_seconds)
    time_limit_seconds : positive integer
        - the time limit in seconds for the solver (by default set to 1 year)
        - after this time, whatever solution is available is returned
        - advisable to set gap and time together, because getting a solution within a given gap can still take a long time,
          so best to cap it.
    max_N_threads : positive integer
        - the maximal number of threads that the solver can use
          more threads = usually faster convergence (but make sure your machine can handle it) 
    
    Returns
    -------
    List of integer indices, of length equal to the number of columns of tasks_vs_clusters_array, indicating
    to which final group (subset) each initial group (cluster) must be reassigned to.
    NOTE: there are len(sizes) unique indices in this list, and it starts at 1, not 0.
    """
    # Calculate the fractions from sizes

    fractional_sizes = sizes / np.sum(sizes)

    S = len(sizes)

    # Normalise the data matrix; if row sums are already 1, this does not change anything
    tasks_vs_clusters_array = tasks_vs_clusters_array / tasks_vs_clusters_array.sum(axis = 1, keepdims = True)

    # Find the number of tasks + compounds row (M) and the number of initial clusters (N)
    M, N = tasks_vs_clusters_array.shape
    if (S > N):
        errormessage = 'The requested number of new clusters to make ('+ str(S) + ') cannot be larger than the initial number of clusters (' + str(N) + '). Please review.'
        raise ValueError(errormessage)

    # Given matrix A (MxN) of fraction of data per cluster, assign each cluster to one of S final ML subsets,
    # so that the fraction of data per ML subset is closest to the corresponding fraction_size.
    # The weights on each ML subset (WML, Sx1) are calculated from fractional_sizes harmonic-mean-like.
    # The weights on each task (WT, Mx1) are calculated as requested by the user.
    # In the end: argmin SUM(ABS((A.X-T).WML).WT)
    # where X is the (NxS) binary solution matrix
    # where T is the (MxS) matrix of target fraction sizes (repeat of fractional_sizes)
    # constraint: assign one cluster to one and only one final ML subset
    # i.e. each row of X must sum to 1

    # Linear programming equivalent: was described elsewhere

    # Copy the input matrix into A, for ease of representation

    A = np.copy(tasks_vs_clusters_array)

    # Create WT = obj_weights

    obj_weights = [1] * M

    if task_weights is not None :
        if len(task_weights) == M :
            obj_weights = task_weights.copy()
        else:
            err = 'The provided task_weights list has a length differing from the number of rows in the input array (' + str(M) + ')'
            raise ValueError(err)

    if equal_weight_perc_compounds_as_tasks == False :
        if M > 1 :
            sum_tw1 = sum(obj_weights[1:])
            obj_weights = [sum_tw1 * obj_weights[0]] + obj_weights[1:]            
    
    obj_weights = obj_weights / np.sum(obj_weights)

    # Create WML
    sk_harmonic = (1 / fractional_sizes) / np.sum(1 / fractional_sizes)

    # Create the pulp model
    prob = LpProblem("Data_balancing", LpMinimize)

    # Create the pulp variables
    # x_names represent clusters,ML_subsets, and are binary variables
    x_names = ['x_'+str(i) for i in range(N * S)]
    x = [LpVariable(x_names[i], lowBound = 0, upBound = 1, cat = 'Integer') for i in range(N * S)]
    # X_names represent tasks,ML_subsets, and are continuous positive variables
    X_names = ['X_'+str(i) for i in range(M * S)]
    X = [LpVariable(X_names[i], lowBound = 0, cat = 'Continuous') for i in range(M * S)]

    # Add the objective to the model

    obj = []
    coeff = []
    for m in range(S):
        for t in range(M):
            obj.append(X[m*M+t])
            coeff.append(sk_harmonic[m] * obj_weights[t])

    prob += LpAffineExpression([(obj[i],coeff[i]) for i in range(len(obj)) ])

    # Add the constraints to the model

    # Constraints forcing each cluster to be in one and only one ML_subset
    for c in range(N):
        prob += LpAffineExpression([(x[c+m*N],+1) for m in range(S)]) == 1

    # Constraints forcing each ML_subset to be non-empty
    for m in range(S):
        prob += LpAffineExpression([(x[i],+1) for i in range(m*N,(m+1)*N)]) >= 1
        
    # Constraints related to the ABS values handling, part 1 and 2
    for m in range(S):
        for t in range(M):
            cs = [c for c in range(N) if A[t,c] != 0]
            prob += LpAffineExpression([(x[c+m*N],A[t,c]) for c in cs]) - X[m*M+t] <= fractional_sizes[m]
            prob += LpAffineExpression([(x[c+m*N],A[t,c]) for c in cs]) + X[m*M+t] >= fractional_sizes[m]

    # Solve the model
    prob.solve(PULP_CBC_CMD(gapRel = relative_gap, timeLimit = time_limit_seconds, threads = max_N_threads))

    # Extract the solution

    list_binary_solution = [value(x[i]) for i in range(N * S)]
    list_initial_cluster_indices = [(list(range(N)) * S)[i] for i,l in enumerate(list_binary_solution) if l == 1]
    list_final_ML_subsets = [(list((1 + np.repeat(range(S), N)).astype('int64')))[i] for i,l in enumerate(list_binary_solution) if l == 1]
    mapping = [x for _, x in sorted(zip(list_initial_cluster_indices, list_final_ML_subsets))]

    return(mapping)


# In[10]:


# 1.5. Function required to evaluate the chemical disjointness of a specific clustering

def min_intercluster_global_distances_and_clusters_Shannon_entropies(clusters, fps, calculate_intercluster_dists = False):
    """
    Calculates the minimal inter-cluster global distances and Shannon entropies of the input clusters, given their fps.
    - 'Global' distances refers to the fact that all the fingerprint bits of the molecules that belong to a cluster are collated
      and reduced to a single 'global' fingerprint, and the distance between two clusters is calculated as the Cosine distance between
      those, rather than between all pairs of *compound* fingerprints. This method was introduced to increase the speed of the calculation.
    - 'Shannon entropy' is defined in the literature as a measure of the information content. A cluster with many fingerprint bits of
      mostly average frequencies will have a higher entropy than one with fewer fingerprint bits / many frequencies close to 0 or 1.
    
    Parameters
    ----------
    clusters : list of integers
        cluster indices of each molecule
    fps : list of rdkit fingerprints
        the fingerprints of each molecule, in the same order as the cluster indices
    calculate_intercluster_dists : bool
        should the inter-cluster distances be calculated, or only the Shannon metrics?
    
    Returns
    -------
    List with:
    0. sorted list of cluster indices (from the input parameter 'clusters')
    1. corresponding list of minimal intercluster global FP Cosine distances
    2. corresponding list of per cluster Shannon entropies (can take any value >= 0)
    3. corresponding list of per cluster normalized Shannon entropies ([0,1])    
    """    
    # Identify the type of fingerprint
    
    allowed_bit_vector_classes = ["<class 'rdkit.DataStructs.cDataStructs.ExplicitBitVect'>",
                                  "<class 'rdkit.DataStructs.cDataStructs.SparseBitVect'>"]
    
    allowed_integer_vector_classes = ["<class 'rdkit.DataStructs.cDataStructs.UIntSparseIntVect'>",
                                     "<class 'rdkit.DataStructs.cDataStructs.ULongSparseIntVect'>",
                                     "<class 'rdkit.DataStructs.cDataStructs.LongSparseIntVect'>",
                                     "<class 'rdkit.DataStructs.cDataStructs.IntSparseIntVect'>"]
    
    if (str(type(fps[0])) in allowed_bit_vector_classes):
        case = "BitVect"
    elif (str(type(fps[0])) in allowed_integer_vector_classes):
        case = "IntVect"
    else:
        raise ValueError("The fingerprints are not any of the allowed bit or integer vector types.")
    
    # Make clusters dictionary and global fingerprint dictionary
    
    cls = dict()
    global_fps = dict()
    global_fps_bitsums = dict()
    
    for i, cli in enumerate(clusters):
        
        fpi = fps[i]
        
        if (case == "BitVect"):
            fpi_bits_list = [fpi.GetOnBits()[j] for j in range(fpi.GetNumOnBits())]
        else:
            fpi_bits_list = list(fpi.GetNonzeroElements().keys())
                
        if cli in cls:
            cls[cli].append(i)
            global_fps[cli].extend(fpi_bits_list)
            global_fps_bitsums[cli] += len(fpi_bits_list)
        else:
            cls[cli] = [i]
            global_fps[cli] = fpi_bits_list
            global_fps_bitsums[cli] = len(fpi_bits_list)
    
    # Also: measure the cluster sizes, i.e. # compounds in each cluster (required for 'new' Shannon entropy calculation)
    
    cls_sizes = dict()
    for cli in cls:
        cls_sizes[cli] = len(cls[cli])
    
    # Turn each element in global_fps into a dictionary of frequencies of bits
        
    for cli in global_fps:
        bits = global_fps[cli]
        global_fps[cli] = dict()
        for b in bits:
            if b in global_fps[cli]:
                global_fps[cli][b] += 1
            else:
                global_fps[cli][b] = 1
    
    cl_Ns = list(np.sort(list(global_fps.keys())))
    min_intercl_global_dist = []
        
    if (calculate_intercluster_dists == True):
    
        # Calculate similarities between clusters global fp's

        for cli in cl_Ns:
            global_fp_i = global_fps[cli]
            N_bits_i = len(global_fp_i)

            current_dists_i = []

            for clj in cl_Ns:
                if cli != clj:
                    global_fp_j = global_fps[clj]
                    N_bits_j = len(global_fp_j)

                    overlap = len(global_fp_i.keys() & global_fp_j.keys())
                    cosine_dist = 1 - overlap / np.sqrt(N_bits_i * N_bits_j)
                    current_dists_i.append(cosine_dist)

            # print(current_dists_i)

            min_intercl_global_dist.append(np.min(current_dists_i))

    # Calculate Shannon entropy per cluster
    
    Shannon = []
    Shannon_norm = []
    
    for cli in cl_Ns:
        global_fp_i = global_fps[cli]
        M = len(global_fp_i)
        N = cls_sizes[cli]
        fi = global_fp_i.values()
        Sg = np.sum([fii * np.log2(fii) for fii in fi])
        Sh = np.sum([(N - fii) * np.log2(N - fii) for fii in fi if fii < N])
        ShEn = M * np.log2(N) - (Sg + Sh) / N
        Shannon.append(ShEn)
        Shannon_norm.append(ShEn / M)
        
    out = [cl_Ns, min_intercl_global_dist, Shannon, Shannon_norm]
    
    return out


# In[11]:


# 1.6. Function that calculates minimal intercluster distances

def min_intercluster_distances(clusters, fps):
    """
    Calculates the minimal Tanimoto distance of each molecule to molecules that are in other clusters.
    This is a measure of how well clustered a set is (many large minimal distances indicate better clustering).
    
    Parameters
    ----------
    clusters : list of integers
        cluster indices of the molecules
    fps : list of rdkit fingerprints
        fingerprints of the molecules, in the same order as the cluster indices
    
    Returns
    -------
    List of the minimal Tanimoto distances of each molecule to molecules that are in other clusters,
    in the same order as the input lists.
    """    
    cls = dict()
    
    for i, cli in enumerate(clusters):
        if cli in cls:
            cls[cli].append(i)
        else:
            cls[cli] = [i]
    
    # print([len(cls[c]) for c in cls])
    
    min_dists = [0] * len(clusters)
    
    for c in cls:
        #print("Comparing cluster",c,"with the rest...")
        cluster = cls[c]
        cfps = [fps[x] for x in cluster]
        ofps = [fps[x] for x in range(len(clusters)) if x not in cluster]
        for i, molid in enumerate(cluster):
            maxicsim = np.max(DataStructs.BulkTanimotoSimilarity(cfps[i],ofps))
            min_dists[molid] = 1 - maxicsim
        
    return min_dists

# Auxiliary data summary function

def make_data_summary_from_pivoted_csv(
    path_to_input_csv,
    subset_column_name = None,
    columns_to_ignore = [],
    add_number_of_records_column = True,
    number_of_records_column_name = "N_records",
    sizes = None
    ):
    """
    Convenience function to create the summary of records/data counts and fractions from a file
    with pivoted data, optionally already clustered/grouped.
    
    Parameters
    ----------
    path_to_input_csv : string
        path to a *pivoted* ((g)zipped) *csv* file with data columns
        - each row is counted as an individual record
        - each column that is not subset_column_name or in columns_to_ignore is interpreted as data
        - all column values that are not NA are counted as data points
    subset_column_name : string, or None
        the name, in the input csv file, of the column identifying the subset each records belongs to
        - if specified, the counts will be given as a pandas DataFrame where each row is a subset and
          each column is named by the data column for which counts and fractions are given;
        - if left to None, the simple count of data per data column will be given, as
          a pandas DataFrame of a single column, where rows are named like the data columns.
    columns_to_ignore : list of strings
        - the list of columns, in the input csv file, that do not constitute data to summarise, and
          must be ignored in the calculation of number of data points.
    add_number_of_records_column : bool
        - if True, a column will be added (and reported as first column after the one indicating the
          subset name, if any), with the number and fraction of total records (regardless of data being there)
        - if False, this will be omitted
    number_of_records_column_name : string
        - only applicable if add_number_of_records_column == True
        - the name of the column reporting the statistics on the records (make sure it hasn't the same
          name as some data column)
    sizes : list of positive numbers, or None
        - the list of target subset sizes, indicating what relative size each subset should be
        - must have the same length as the number of unique values in subset_column_name column
        - must be ordered so it matches the *sorted* *unique* values in subset_column_name column
        - example: if subset_column_name contains [3, 1, 2, 3, 2...] --> unique [3, 1, 2] -->
          sorted [1, 2, 3] : then sizes can be e.g. [10, 50, 40], meaning that subset 1 should have
          fractional size 10/(10+50+40), subset 2 50/(10+50+40), subset 3 40/(10+50+40)
        - this is used to calculate the weighted sum of absolute differences (see Returns)
    
    Returns
    -------
    List with:
    0. numpy array reporting the number of records/data points per data column (--> columns), per subset (--> rows)
    1. numpy array like 0 but reporting the fractions instead of the counts (i.e. each column is divided by its column sum)
    2. weighted sum of absolute differences between the expected and observed fractions of records/data points
       (only if subset_column_name and sizes are both != None).
       Example
       If sizes = [1, 2, 2], it is expected that 3 subsets are present, with fractional sizes [0.2, 0.4, 0.4], respectively.
       The absolute difference between these sizes and each column in the fractions array is computed.
       Harmonic weighting is applied, and the resulting numbers are summed.
       
       Fractions:
       subset N_recs A     B     C
       1      0.21   0.20  0.15  0.25
       2      0.40   0.35  0.50  0.35
       3      0.39   0.45  0.35  0.40

       Column-wise absolute differences from expected fractions [0.2, 0.4, 0.4]:
       subset N_recs A     B     C
       1      0.01   0.00  0.05  0.05
       2      0.00   0.05  0.10  0.05
       3      0.01   0.05  0.05  0.00
       
       Apply harmonic weights = [1/0.2, 1/0.4, 1/0.4] / (1/0.2 + 1/0.4 + 1/0.4) = [0.5, 0.25, 0.25]
       (this balances the importance of each row, so the differences for the rows with higher expected fractions do not override smaller ones)
       
       subset N_recs  A       B      C
       1      0.005   0.000   0.025  0.025
       2      0.000   0.0125  0.025  0.0125
       3      0.0025  0.0125  0.0125 0.000
       
       (note that difference 0.05 for B.1 (expected fraction 0.2), now counts the same as 0.10 for B.2 (expected fraction 0.4)
       
       The sum of all the numbers in the matrix is the final output (for this example, 0.1325).       
    """    
    # BEGIN PROCESSING

    print("Reading the compound ID's and SMILES from the csv file...")
    print("")

    # debug: this made a very large df, which can instead be made sparse
    #df = pd.read_csv(path_to_input_csv)
    # Use instead a function that reads columns in chunks and makes a dataframe with sparse columns

    if subset_column_name is not None:
        df = read_csv_to_dataframe_with_sparse_cols(
            dense_columns_names = [subset_column_name],
            input_csv_file_full_path = path_to_input_csv)
    else:
        df = read_csv_to_dataframe_with_sparse_cols(
            dense_columns_names = [],
            input_csv_file_full_path = path_to_input_csv)
        subset_column_name = "_temp_subset_column_name"
        df[subset_column_name] = np.repeat("ALL_RECORDS", df.shape[0])
        
    # remove the columns to ignore    
    df.drop(columns = columns_to_ignore, inplace = True, errors = 'ignore')

    # Unpivot the data, for making the summary faster
    print("Unpivoting the data columns...")
    print("")
    
    cols_with_data = df.columns.values[~(df.columns.isin([subset_column_name]))]

    df_dataonly_unpivoted = unpivot_dataframe(
        dataframe = df,
        ID_column_name = subset_column_name,
        data_columns_names = cols_with_data,
        property_column_name = "task_name",
        value_column_name = None)    
    
    # Create the data summary
    
    # Use cross tabulation to make the array of counts of data per cluster per task
    data_count = pd.crosstab(index = df_dataonly_unpivoted[subset_column_name], columns = df_dataonly_unpivoted["task_name"])

    col = data_count.pop(subset_column_name)
    
    if (add_number_of_records_column == True):
        data_count = pd.concat([col, data_count], axis = 1, ignore_index = False)
        data_count.rename({subset_column_name : number_of_records_column_name}, axis = 1, inplace = True)

    data_percentages = data_count.transform(lambda x: x / x.sum())
        
    # Calculate the weighted sum of absolute differences, for reporting
    
    if (subset_column_name is not None) & (sizes is not None):    
        fractional_sizes = sizes / np.sum(sizes)
        sk_harmonic = (1 / fractional_sizes) / np.sum(1 / fractional_sizes)
        absdiffs = np.abs(data_percentages - np.array([fractional_sizes]).transpose())
        wsabsdiffs = np.dot(np.atleast_2d(sk_harmonic), absdiffs).sum()
    else:
        wsabsdiffs = 0
    
    return [data_count, data_percentages, wsabsdiffs]

# Data balancing function starting from pre-clustered csv file

def balance_data_from_csv_file_pulp(path_to_input_csv,
                                    path_to_output_csv,
                                    initial_subset_column_name,
                                    sizes,
                                    columns_to_ignore = [],
                                    final_subset_column_name = "ML_subset",
                                    equal_weight_perc_compounds_as_tasks = False,
                                    balance_categorical_labels = False,
                                    balance_continuous_distributions = False,
                                    interpret_censored_floats_as_floats = True,
                                    N_bins = 5,
                                    relative_gap = 0,
                                    time_limit_seconds = 60*60*24*365,                                    
                                    max_N_threads = 4):
    """
    Takes as input a *pivoted* csv file with initial subset (cluster) identifiers and tasks (properties to model) data, possibly sparse.
    Merges clusters into the required number of final subsets, at the same time 'balancing' the data, i.e.:
      > in general: the number of records (=distinct SMILES) should be similar among all final subsets
      > for all types of data: for each task, the number of data points should be similar among all final subsets
      > for continuous data: for each task, the distributions of values should be similar among all final subsets
      > for categorical data: for each task, the % of data belonging each category should be similar among all final subsets
      
    Parameters
    ----------
    path_to_input_csv : string
        path to *pivoted* ((g)zipped) csv file with initial subset identifiers and tasks (properties to model) data, possibly sparse.
    path_to_output_csv : string
        path to *pivoted* ((g)zipped) csv output file
        - this will be the input file with additional column (overwriting if it exists in the input):
          > final_subset_column_name : the final subset to which the record belongs
            NOTE: if sizes has length n, the values of final_subset_column_name are 1 .. n
    initial_subset_column_name : string
        the name, in the input csv file, of the column identifying the *initial* subset or cluster each record belongs to.
    columns_to_ignore : list of strings or empty list
        the names of the columns, in the input file, which are neither the initial_subset_column_name nor relevant data
    sizes : list of positive numbers
        list of the desired final sizes (will be normalised to fractions internally)
    final_subset_column_name : string
        the name, in the output csv file, of the column identifying the *final* subset or cluster each record belongs to.
        - NOTE: make sure it hasn't the same name as some other existing column, or it will be overwritten.
    equal_weight_perc_compounds_as_tasks : bool
        - if True, matching the % records will have the same weight as matching the % data of individual tasks
        - if False, matching the % records will have a weight X times as large as matching the % data of the X tasks
    balance_categorical_labels : bool
        - if True, any column that has even just one non-'float-able' value will be considered categorical,
          > i.e. each distinct value will be considered a category (label), and balanced across the ML subsets.
          > E.g. if a column contains 'good' and 'bad' labels, in 30:70 proportions in the overall dataset,
            it will be attempted to have 30:70 good:bad also in all ML subsets (not only the correct # of data points).
        - Caution: if you have a column with many continuous values and even just one text value included by mistake,
          all distinct numerical values will be seen as a separate category, and probably crash the LP solver.
        - NOTE: sometimes binary categories are encoded as integers, like 0 and 1.
          > The 'correct' option would be to turn them into non-float-able text, like 'inactive' and 'active'.
          > An easier way around is to set balance_continuous_distributions = True with N_bins = 2. Same final effect.
    balance_continuous_distributions : bool
        - if True, any column that has *all* 'float-able' values will be binned into the desired N_bins (below),
          and it will be attempted to reproduce the same distribution in all ML subsets.
          > E.g. if a property 'Y' has bins (-5,-4], (-4,-3], ..., (4, 5], with data in proportions 3:7:...:4,
            it will be attempted to have the same bins and proportions in all ML_subsets.
    interpret_censored_floats_as_floats : bool
        - if True, values like '<5' or '>100' will be intepreted as floats (5 and 100) *for distribution balancing purposes*
          > meaning: the original data will still be saved as they are in the input, but the distribution calculations will use uncensored data
          > in addition, for convenience, an output file with uncensored data will be saved, too (only for columns that become fully continuous by uncensoring)
        - if False, such values will be interpreted as text (and the presence of even just one of them will make a column categorical!)
    N_bins : positive integer
        - only applicable if balance_continuous_distributions == True
        - the number of bins to make for the distribution of continuous variables to be balanced across subsets
        - Caution: 5 bins means that the LP variables to solve for are ~multiplied by 5.
          This may result in a *much* longer run time. Use only if strictly necessary / if you have seen distribution unbalance.
        - Technical note on binning:
          > pandas.cut is used, with ~equally sized intervals in the found range of continuous data.
          > Exception: when there are left or right outliers (<= Q1 - 1.5 * IQR or >= Q3 - 1.5 * IQR), those are not
            included in the calculation of the intervals, and they are added back in as lower and upper limits.
          > E.g. [-100, -1, 0, 1, 3, 9], request 5 bins --> -100 is outlying --> range 9-(-1) = 10 --> intervals of 10/5 = 2 units
            --> [-1, 1], (1, 3], (3, 5], (5, 7], (7, 9] --> [-100, 1], (1, 3], (3, 5], (5, 7], (7, 9]
    > Parameters of the linear programming solver
        relative_gap
        time_limit_seconds
        max_N_threads
        - see documentation of function 'balance_data_from_tasks_vs_clusters_array_pulp' for details
    
    Returns
    -------
    List with:
    0. numpy array reporting the number of records/data points per data column (--> columns), per subset (--> rows)
    1. numpy array like 0 but reporting the fractions instead of the counts (i.e. each column is divided by its column sum)
    2. weighted sum of absolute differences between the expected and observed fractions of records/data points
       (only if subset_column_name and sizes are both != None).
       Example
       If sizes = [1, 2, 2], it is expected that 3 subsets are present, with fractional sizes [0.2, 0.4, 0.4], respectively.
       The absolute difference between these sizes and each column in the fractions array is computed.
       Harmonic weighting is applied, and the resulting numbers are summed.
       
       Fractions:
       subset N_recs A     B     C
       1      0.21   0.20  0.15  0.25
       2      0.40   0.35  0.50  0.35
       3      0.39   0.45  0.35  0.40

       Column-wise absolute differences from expected fractions [0.2, 0.4, 0.4]:
       subset N_recs A     B     C
       1      0.01   0.00  0.05  0.05
       2      0.00   0.05  0.10  0.05
       3      0.01   0.05  0.05  0.00
       
       Apply harmonic weights = [1/0.2, 1/0.4, 1/0.4] / (1/0.2 + 1/0.4 + 1/0.4) = [0.5, 0.25, 0.25]
       (this balances the importance of each row, so the differences for the rows with higher expected fractions do not override smaller ones)
       
       subset N_recs  A       B      C
       1      0.005   0.000   0.025  0.025
       2      0.000   0.0125  0.025  0.0125
       3      0.0025  0.0125  0.0125 0.000
       
       (note that difference 0.05 for B.1 (expected fraction 0.2), now counts the same as 0.10 for B.2 (expected fraction 0.4)
       
       The sum of all the numbers in the matrix is the final output (for this example, 0.1325).       
        
    Saves:
    - a final output csv file with the same data as the input file, plus columns with:
      - the identifier of the final ML subset 
      - optionally, the inter-subset distances
      File name : path_to_output_csv
    - if interpret_censored_floats_as_floats == True:
      > a file identical to path_to_output_csv BUT with censored data uncensored (e.g. with '>5' turned to 5)
        (only for columns that become fully continuous when uncensoring is applied)
        File name : path_to_output_csv with '.csv' replaced by '_uncensored.csv'    
    
    Optionally saves data summary files, to document the data balancing performance.    
    """
    # START PROCESSING
    
    # Read the data
    # debug: this made a very large df, which can instead be made sparse
    #df = pd.read_csv(path_to_input_csv)
    # Use instead a function that reads columns in chunks and makes a dataframe with sparse columns

    df = read_csv_to_dataframe_with_sparse_cols(
        dense_columns_names = [initial_subset_column_name],
        input_csv_file_full_path = path_to_input_csv)
    
    # Make the initial data summary to submit to the data balancing function

    # Create the lists of relevant column names
    cols_with_data_and_ID = df.columns.values[~(df.columns.isin(columns_to_ignore))]
    #columns_to_ignore.append(number_of_records_column_name)
    cols_with_data = df.columns.values[~(df.columns.isin(columns_to_ignore + [initial_subset_column_name]))]
    # Create the intermediate data summary before balancing
    # The following data summary method no longer works for a sparse DataFrame...
    #data_count = df[cols_for_counting].groupby([initial_subset_column_name]).count()
    #data_percentages = data_count.transform(lambda x: x / x.sum())
    #tasks_vs_clusters_array = np.array(data_percentages, ndmin = 2).transpose()
    # Replaced by a different method based on the unpivoted data

    # Unpivot the data, for making the summary faster
    print("Preparing for cluster merging + balancing")
    print("")

    # New: now we allow to balance labels and/or distributions
    # --> this is achieved by splitting task names by label or bin
    # --> this requires specific weights for tasks, otherwise a task that has many labels or bins is over-weighted
    # In this case, the values must be retained in the unpivoted dataset, for later use

    if ((balance_categorical_labels == True) or (balance_continuous_distributions == True) or (interpret_censored_floats_as_floats == True)):
        value_column_name = "value_name"
        # Define a function that returns an uncensored float from a value, otherwise '', optionally trying uncensoring
        # NOTE: inf needs to be considered valid floats (NA isn't; but in unpivoted data as per output of unpivot_dataframe there should be no NA's)
        def float_unc(v, try_uncensoring):
            try:
                v = v.lstrip()
            except:
                pass
            u = ''
            try:
                u = float(v)
            except:
                if try_uncensoring == True:
                    try:
                        v = str(v)
                        uf = float(v[1:])
                        cf = v[0]
                        if ((cf == '<') | (cf == '>')):
                            u = uf
                    except:
                        pass
            return u
        # Define a function that takes a list of values and makes outlier-robust cuts for pd.cut
        def make_robust_cuts(s, N_bins):
            Q1 = np.quantile(s, 0.25)
            Q3 = np.quantile(s, 0.75)
            IQR = Q3 - Q1
            if IQR != 0:
                lol = Q1 - 1.5 * IQR
                uol = Q3 + 1.5 * IQR
                s_red = [n for n in s if ((n > lol) & (n < uol))]
            else:
                s_red = s
            r = max(s_red) - min(s_red)
            step = r / N_bins
            cuts = [min(s_red) + i * step for i in range(N_bins + 1)]
            cuts[0] = min(s)
            cuts[-1] = max(s)
            # if cuts only contains 1 unique value, pd.cut will return nan --> correct to 1 (= pd.cut will make 1 bin)
            if len(pd.unique(cuts)) == 1 :
                cuts = 1
            return cuts
    else:
        value_column_name = None

    print("   Unpivoting the data columns...")
    print("")
    df_dataonly_unpivoted = unpivot_dataframe(
        dataframe = df[cols_with_data_and_ID],
        ID_column_name = initial_subset_column_name,
        data_columns_names = cols_with_data,
        property_column_name = "task_name",
        value_column_name = value_column_name)

    print("   Indexing the tasks...")
    print("")
    # Collect the index of task data for each task_name, and store in a dict    
    unique_task_names = pd.unique(df_dataonly_unpivoted["task_name"])
    index_vs_task_name_dict = dict()
    type_vs_task_name_dict = dict()
    for tn in unique_task_names:
        task_index = df_dataonly_unpivoted[df_dataonly_unpivoted['task_name'] == tn].index
        index_vs_task_name_dict[tn] = task_index
        # Provisionally set all tasks to categorical
        type_vs_task_name_dict[tn] = 'cat'

    # Create the required weights dictionary,
    # by identifying the number of bins of each task (categorical or continuous) when necessary
    # or setting to 1 as default when no balancing of labels or distributions is required
    weights_for_tasks = dict()

    # Identify which tasks are continuous, taking into account the setting of interpret_censored_floats_as_floats
    # - obviously this is only applicable if there are categorical labels or continuous distributions to balance
    if ((balance_categorical_labels == True) or (balance_continuous_distributions == True) or (interpret_censored_floats_as_floats == True)):
        print("   Classifying tasks into strictly continuous and categorical...")
        print("")        
    
    for tn in unique_task_names:
        weights_for_tasks[tn] = 1
        if tn != initial_subset_column_name :
            # If balancing categorical labels and/or continuous distributions is required, split the 'task_name' accordingly
            if ((balance_categorical_labels == True) or (balance_continuous_distributions == True) or (interpret_censored_floats_as_floats == True)):
                # Find the values for the task
                task_index = index_vs_task_name_dict[tn]
                vals = df_dataonly_unpivoted.loc[task_index, value_column_name].copy()
                # Apply float_unc to the values
                vals_u = vals.apply(lambda v: float_unc(v, try_uncensoring = interpret_censored_floats_as_floats))
                # Decide if the values were all float-able
                vals_are_all_numerical = False
                if all(vals_u != ''):
                    vals_are_all_numerical = True
                    # If so, store the knowledge that this task is continuous into type_vs_task_name_dict                    
                    type_vs_task_name_dict[tn] = 'cont'
                    # If it was required to interpret censored values as floats, replace vals with vals_u
                    if interpret_censored_floats_as_floats == True:
                        #df_dataonly_unpivoted.loc[task_index, value_column_name] = list(vals_u.values)
                        vals = vals_u.copy()
                # For tasks that have at least one non-numerical value, if required to balance the labels,
                # rename task_name's according to the values
                if ((balance_categorical_labels == True) & (vals_are_all_numerical == False)) :                    
                    split_tasks = [str(tn) + '_(' + str(v) + ')' for v in vals]
                    df_dataonly_unpivoted.loc[task_index, 'task_name'] = split_tasks
                    unique_vals = pd.unique(vals)
                    split_weight = 1 / len(unique_vals)
                    for stn in pd.unique(split_tasks) :
                        weights_for_tasks[stn] = split_weight                    
                    print("    > Splitting categorical property '" + str(tn) + "' into " + str(len(unique_vals)) + ' columns...')
                # For tasks that have only numerical values, if required to balance data distributions,
                # bin the vals and rename task_name's according to the bins
                elif ((balance_continuous_distributions == True) & (vals_are_all_numerical == True)) :                    
                    cuts = make_robust_cuts(list(vals), N_bins)
                    vals_bins = pd.cut(vals, bins = cuts, include_lowest = True, duplicates = 'drop')
                    split_tasks = [str(tn) + '_' + str(b) for b in vals_bins]
                    df_dataonly_unpivoted.loc[task_index, 'task_name'] = split_tasks
                    unique_vals_bins = pd.unique(vals_bins)
                    split_weight = 1 / len(unique_vals_bins)
                    for stn in pd.unique(split_tasks) :
                        weights_for_tasks[stn] = split_weight                    
                    print("    > Splitting continuous property '" + str(tn) + "' into " + str(len(unique_vals_bins)) + ' columns...')

    print("   Making data summaries per task, per initial cluster...")
    print("")

    # Use cross tabulation to make the array of counts of data per cluster per task
    data_count_per_initial_cl = pd.crosstab(index = df_dataonly_unpivoted[initial_subset_column_name], columns = df_dataonly_unpivoted["task_name"])
    # VERY IMPORTANT: the records count column must be the first!
    col = data_count_per_initial_cl.pop(initial_subset_column_name)
    data_count_per_initial_cl = pd.concat([col, data_count_per_initial_cl], axis = 1, ignore_index = False)
    data_percentages = data_count_per_initial_cl.transform(lambda x: x / x.sum())
    weights_for_LP = [weights_for_tasks[tn] for tn in list(data_count_per_initial_cl.columns)]
    # Transpose the array, for use in the LP solver function
    tasks_vs_clusters_array = np.array(data_count_per_initial_cl, ndmin = 2).transpose()
    
    print("Balancing the data (this may take a long time)...")
    print("")

    print("   Starting the linear program solver...")
    print("")

    # Run the balancing function
    mapping = balance_data_from_tasks_vs_clusters_array_pulp(
        tasks_vs_clusters_array = tasks_vs_clusters_array,
        sizes = sizes,
        task_weights = weights_for_LP,
        equal_weight_perc_compounds_as_tasks = equal_weight_perc_compounds_as_tasks,
        relative_gap = relative_gap,
        time_limit_seconds = time_limit_seconds,
        max_N_threads = max_N_threads)

    if (len(mapping) < tasks_vs_clusters_array.shape[1]):
        raise ValueError("The linear program solver did not reach a feasible solution within the specified time limit. \
            Please increase it.")

    print("Balancing completed.")
    print("")

    # Store in df to which final set each initial subset maps to
    initial_subset_values = list(data_percentages.index.values)
    final_subset_vs_initial_subset_dict = dict(zip(initial_subset_values, mapping))
    df[final_subset_column_name] = [final_subset_vs_initial_subset_dict[i] for i in df[initial_subset_column_name]]
    
    # Save the results
    # No longer imposing a hard-coded output path, now a parameter in the function
    #path_to_output_csv = path_to_input_csv.replace(".csv", "_balanced.csv")
    print("Saving main output csv file in ",path_to_output_csv," ...")
    print("") 
    # # The use of sparse DataFrame columns causes horrible slowness in to_csv.
    # To counteract that, the file is written out in chunks, using an appropriately defined function.   
    #df.to_csv(path_to_output_csv, index = False)       
    write_csv_from_dataframe_with_sparse_cols(
        dataframe = df,
        sparse_columns_names = cols_with_data,
        output_csv_file_full_path = path_to_output_csv)
    
    if interpret_censored_floats_as_floats == True:
        # Uncensor in df all tasks that were found to be fully continuous
        for tn in unique_task_names:        
            if tn != initial_subset_column_name :
                if type_vs_task_name_dict[tn] == 'cont':                
                    vals_u = df[tn].apply(lambda v: float_unc(v, True))
                    df[tn] = pd.arrays.SparseArray([u if u != '' else pd.NA for u in vals_u])
        path_to_output_csv_uncensored = path_to_output_csv.replace('.csv', '_uncensored.csv')
        print("Saving main *uncensored* output csv file in ",path_to_output_csv_uncensored," ...")
        print("")
        # The use of sparse DataFrame columns causes horrible slowness in to_csv.
        # To counteract that, the file is written out in chunks, using an appropriately defined function.
        #df.to_csv(path_to_output_csv, index = False)
        write_csv_from_dataframe_with_sparse_cols(
            dataframe = df,
            sparse_columns_names = cols_with_data,
            output_csv_file_full_path = path_to_output_csv_uncensored)

    # Make the final data summary and report it

    # Store in df_dataonly_unpivoted to which final set each initial subset maps to
    df_dataonly_unpivoted[final_subset_column_name] = [final_subset_vs_initial_subset_dict[i] for i in df_dataonly_unpivoted[initial_subset_column_name]]
    
    # Use cross tabulation to make the array of counts of data per cluster per task
    data_count_per_final_subset = pd.crosstab(index = df_dataonly_unpivoted[final_subset_column_name], columns = df_dataonly_unpivoted["task_name"])
    # VERY IMPORTANT: the records count column must be the first!
    col = data_count_per_final_subset.pop(initial_subset_column_name)
    col.name = final_subset_column_name
    data_count_per_final_subset = pd.concat([col, data_count_per_final_subset], axis = 1, ignore_index = False)
    data_percentages = data_count_per_final_subset.transform(lambda x: x / x.sum())

    fractional_sizes = sizes / np.sum(sizes)
    sk_harmonic = (1 / fractional_sizes) / np.sum(1 / fractional_sizes)
    absdiffs = np.abs(data_percentages - np.array([fractional_sizes]).transpose())
    wsabsdiffs = np.dot(np.atleast_2d(sk_harmonic), absdiffs).sum()
    
    return [data_count_per_final_subset, data_percentages, wsabsdiffs]

# Auxiliary function to obtain the 'best' chemical split (the one with the maximal median minimal inter-cluster distance)

def make_best_chemically_disjoint_clusters(
    path_to_input_csv,
    path_to_output_csv,    
    initial_cluster_column_name = "initial_cluster",
    smiles_column_name = "SMILES",
    folded_to_BitVect = False,
    morgan_radius = 3,
    morgan_NBits = 32768,
    reduced_NBits_for_optim = 512,
    initial_clustering_method = "sphere exclusion",    
    calculate_min_inter_cluster_TanimotoDists = True,
    min_intercluster_distance_column_name = "min_inter_cluster_Tanimoto_distance",
    min_dist = [0.7, 0.9],
    assign_to_initial_centers_only = True,
    similarity_aggregation = 'mean',
    seed = -1,
    priority_to_removal_of_bits_from_S0 = False,
    min_overlap = 4,
    min_sim = [0.5, 0.7]):
    """
    Takes as input a csv file with SMILES.    
    Clusters the records by chemistry, using either the 'sphere exclusion' method, or the 'iterative minimal overlap' method.
    NOTE: the best parameter values for clustering are automatically searched by this function.
    
    Parameters
    ----------
    path_to_input_csv : string
        path to *pivoted* ((g)zipped) csv file with SMILES.
    path_to_output_csv : string
        path to *pivoted* ((g)zipped) csv output file
        - this will be the input file with additional column initial_cluster_column_name and, if required, min_intercluster_distance_column_name
    initial_cluster_column_name : string
        the name of the column, in the output csv, indicating the cluster each record was assigned to.
        NOTE: if N clusters are made, the values of initial_cluster_column_name are 0 .. (n-1)
    smiles_column_name : string
        name of the column containing the SMILES in the input csv
    folded_to_BitVect : bool
        must the fingerprints be calculated as folded bit vectors?
        - if True, the calculations will be faster but the results less accurate
        - if False, full fingerprints will be used, yielding more accurate results
    morgan_radius : integer {1, 2, 3}
        the radius for the morgan fingerprint calculation, where applicable
    morgan_NBits : positive integer
        the number of bits for the folded fingerprints (only applicable when folded_to_BitVect == True)
    reduced_NBits_for_optim : positive integer or None
        - if None, the search for the best clustering will be done using the fingerprints calculated as per above parameters
        - if numeric, the search for the best clustering will be done using folded fingerprints to this specified number of bits,
          mostly for speed, and once the best parameter is found, the clustering is rerun with the 'full' fingerprints
    initial_clustering_method : string {"sphere exclusion", "iterative min overlap"}
        the method used to do the clustering
        - "sphere exclusion" is faster and only requires one parameter (the minimal MaxMin distance)
        - "iterative min overlap" is slower, but generally achieves a better chemical separation
    calculate_min_inter_cluster_TanimotoDists : bool
        must inter-cluster distances be measured?
        - NOTE: this calculation requires a very long time, so do it only if strictly necessary
    min_intercluster_distance_column_name : string        
        the minimal Tanimoto distance of the compound to any compound in any other clusters (if calculate_min_inter_cluster_TanimotoDists == True).
    > Parameters applicable when initial_clustering_method == "sphere exclusion"
        min_dist : list of 2 floats between 0 and 1
            the boundaries for the automated search of the best min_dist, e.g. [0.6, 0.9]
        assign_to_initial_centers_only : bool
        similarity_aggregation : string {'max', 'mean', 'median'}
        seed : integer
        - see documentation of function 'sphere_exclusion_clustering' for details
    > Parameters applicable when initial_clustering_method == "iterative min overlap"
        priority_to_removal_of_bits_from_S0 : bool
        min_overlap : integer >= 0
        min_sim : list of 2 floats between 0 and 1
            the boundaries for the automated search of the best min_sim, e.g. [0.5, 0.7]
        - see documentation of function 'iterative_clustering_by_minimal_overlap' for details

    Returns
    -------
    None.
    The output file is silently written out.
    """
    # START PROCESSING
    
    # Check first if the required parameters are specified

    if (initial_clustering_method == "sphere exclusion"):
        if (type(min_dist) is list):
            pass
        else:
            raise ValueError("You selected sphere exclusion clustering. You need to specify min_dist as a list of 2 values between 0 and 1.")
        if (len(min_dist) == 2):
            pass
        else:
            raise ValueError("You selected sphere exclusion clustering. You need to specify min_dist as a list of 2 values between 0 and 1.")
        if ((min_dist[0] >= 0) & (min_dist[1] <= 1) & (min_dist[0] < min_dist[1])):
            pass
        else:                
            raise ValueError("You selected sphere exclusion clustering. You need to specify min_dist as a list of 2 values between 0 and 1.")
    elif (initial_clustering_method == "iterative min overlap"):            
        if (type(min_sim) is list):
            pass
        else:
            raise ValueError("You selected iterative min overlap clustering. You need to specify min_sim as a list of 2 values between 0 and 1.")
        if (len(min_sim) == 2):
            pass
        else:
            raise ValueError("You selected iterative min overlap clustering. You need to specify min_sim as a list of 2 values between 0 and 1.")
        if ((min_sim[0] >= 0) & (min_sim[1] <= 1) & (min_sim[0] < min_sim[1])):
            pass
        else:
            raise ValueError("You selected iterative min overlap clustering. You need to specify min_sim as a list of 2 values between 0 and 1.")
    else:
        raise ValueError("initial_clustering_method can only be either 'sphere exclusion' or 'iterative min overlap'.")

    # Begin
    
    print("Find best clustering")
    print("====================")
    print("")
    
    # 1. Read data in csv format (file with smiles)
    print("Reading csv file...")
    print("")
    # debug: this made a very large df, which can instead be made sparse
    #df = pd.read_csv(path_to_input_csv)
    # Use instead a function that reads columns in chunks and makes a dataframe with sparse columns

    df = read_csv_to_dataframe_with_sparse_cols(
        dense_columns_names = [smiles_column_name],
        input_csv_file_full_path = path_to_input_csv)
    
    #cols_with_data = df.columns.values[~(df.columns.isin([smiles_column_name]))]
    # small modification made on 20220127:
    # ensure that cols_with_data does not include any of the dense columns that are created by this process
    # (which might happen if the input file wrongly contains columns with those names, that are interpreted as data)
    cols_with_data = df.columns.values[~(df.columns.isin([smiles_column_name, initial_cluster_column_name, min_intercluster_distance_column_name]))]

    # 2,3. Convert the list of smiles to a list of fingerprints without storing molecules.
    print("Creating fingerprints from SMILES...")
    print("")

    fps = []
    fps_reduced = []

    # New in v20.2 : we use a Morgan FP generator, which must be initialised here
    if (folded_to_BitVect == True) :
        mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius = morgan_radius, fpSize = morgan_NBits)
    else :
        mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius = morgan_radius)
        # Not necessary to get the Bit Info Mapping, if we assume that GetOnBits() is consistent for both types of FP.
        #ao = rdFingerprintGenerator.AdditionalOutput()
        #ao.AllocateBitInfoMap()
    if (reduced_NBits_for_optim is not None):
        mfpgen_red = rdFingerprintGenerator.GetMorganGenerator(radius = morgan_radius, fpSize = reduced_NBits_for_optim)

    for s in df[smiles_column_name]:

        m = Chem.MolFromSmiles(s)

        if (folded_to_BitVect == True):
            #fps.append(rdMolDescriptors.GetMorganFingerprintAsBitVect(m, morgan_radius, morgan_NBits))
            fps.append(mfpgen.GetFingerprint(m))
        else:
            #fps.append(Chem.rdMolDescriptors.GetMorganFingerprint(m, morgan_radius, useCounts = False))
            fps.append(mfpgen.GetSparseFingerprint(m))

        if (reduced_NBits_for_optim is not None):
            #fps_reduced.append(rdMolDescriptors.GetMorganFingerprintAsBitVect(m, morgan_radius, reduced_NBits_for_optim))
            fps_reduced.append(mfpgen_red.GetFingerprint(m))

    if (reduced_NBits_for_optim is not None):
        fps_for_optim = fps_reduced
    else:
        fps_for_optim = fps
    
    # 4. Make the clusters and final subsets
    
    print("Starting selections...")
    print("")
    
    # 4.2. Make the initial clusters by auto-search
        
    print("Started automated search of best initial clustering...")
    print("")

    if (initial_clustering_method == "sphere exclusion"):
        def cl_fun(md):
            cl = sphere_exclusion_clustering(list_of_molecules = None,
                    list_of_fingerprints = fps_for_optim,
                    N_centers_to_pick = None,
                    min_dist = md,
                    assign_to_initial_centers_only = assign_to_initial_centers_only,
                    similarity_aggregation = similarity_aggregation,
                    folded_to_BitVect = folded_to_BitVect,
                    morgan_radius = morgan_radius,
                    morgan_NBits = morgan_NBits,
                    seed = 5)
            print("   Clustering done. Calculating global Cosine inter-cluster distances...")
            out = min_intercluster_global_distances_and_clusters_Shannon_entropies(cl, fps_for_optim, calculate_intercluster_dists = True)
            median_intercluster_global_Cosine_dist = np.median(out[1])
            return [median_intercluster_global_Cosine_dist, cl]

        xl = min_dist[0]
        xr = min_dist[1]

    elif (initial_clustering_method == "iterative min overlap"):
        def cl_fun(msim):
            cl = iterative_clustering_by_minimal_overlap(list_of_molecules = None,
                   list_of_fingerprints = fps_for_optim,
                   priority_to_removal_of_bits_from_S0 = priority_to_removal_of_bits_from_S0,
                   min_overlap = min_overlap,
                   min_sim = msim,
                   folded_to_BitVect = folded_to_BitVect,
                   morgan_radius = morgan_radius,
                   morgan_NBits = morgan_NBits)[0]
            print("   Clustering done. Calculating global Cosine inter-cluster distances...")
            out = min_intercluster_global_distances_and_clusters_Shannon_entropies(cl, fps_for_optim, calculate_intercluster_dists = True)
            median_intercluster_global_Cosine_dist = np.median(out[1])
            return [median_intercluster_global_Cosine_dist, cl]

        xl = min_sim[0]
        xr = min_sim[1]

    min_dist_min = xl
    min_dist_max = xr

    delta = xr - xl
    tol = 0.01

    min_intercluster_cosine_dists_dict = dict()
    tried_clusterings_dict = dict()

    while (True):

        xm = xl + delta / 2

        print("xl,xm,xr = ",(xl,xm,xr))

        if xl not in min_intercluster_cosine_dists_dict:
            print("trying parameter =",xl)
            cl_out = cl_fun(xl)
            yl = cl_out[0]
            min_intercluster_cosine_dists_dict[xl] = yl
            tried_clusterings_dict[xl] = cl_out[1]
        else:
            yl = min_intercluster_cosine_dists_dict[xl]

        if xm not in min_intercluster_cosine_dists_dict:
            print("trying parameter =",xm)
            cl_out = cl_fun(xm)
            ym = cl_out[0]
            min_intercluster_cosine_dists_dict[xm] = ym
            tried_clusterings_dict[xm] = cl_out[1]
        else:
            ym = min_intercluster_cosine_dists_dict[xm]

        if xr not in min_intercluster_cosine_dists_dict:
            print("trying parameter =",xr)
            cl_out = cl_fun(xr)
            yr = cl_out[0]
            min_intercluster_cosine_dists_dict[xr] = yr
            tried_clusterings_dict[xr] = cl_out[1]
        else:
            yr = min_intercluster_cosine_dists_dict[xr]

        if (delta <= tol):
            max_d = max([yl,ym,yr])
            xbest = [xl,xm,xr][[yl,ym,yr].index(max_d)]
            break

        delta = delta / 2

        if ((yl <= ym) & (ym <= yr)):
            xr_new = min(min_dist_max, xr + delta)
            xl_new = xr_new - delta
            xl = xl_new
            xr = xr_new        
        elif ((yl >= ym) & (ym >= yr)):
            xl_new = max(min_dist_min, xl - delta)
            xr_new = xl_new + delta
            xl = xl_new
            xr = xr_new        
        else:
            xl = xm - delta / 2
            xr = xm + delta / 2

    xs = list(min_intercluster_cosine_dists_dict.keys())
    ys = list(min_intercluster_cosine_dists_dict.values())
    index_of_best = ys.index(max(ys))
    xbest = xs[index_of_best]
    max_d = ys[index_of_best]
    print("")
    print("best parameter found =",xbest, ", median inter-cluster Cosine distance =", max_d)
    print("")
    
    ordered = [x for _,x in sorted(zip(xs, range(len(xs))))]
    xs_ordered = [xs[i] for i in ordered]
    ys_ordered = [ys[i] for i in ordered]
    plt.plot(xs_ordered, ys_ordered)
    plt.xlabel('trial parameter')
    plt.ylabel('median inter-cluster global Cosine distance')
    plt.show()
    xys = pd.DataFrame({'trial parameter' : xs_ordered, 'median inter-cluster global Cosine distance' : ys_ordered})
    print(xys)
    print("")

    if reduced_NBits_for_optim is not None:
    
        print("Redoing clustering with best parameter and full fingerprints...")
        print("")

        if (initial_clustering_method == "sphere exclusion"):
            cls = sphere_exclusion_clustering(list_of_molecules = None,
                    list_of_fingerprints = fps,
                    N_centers_to_pick = None,
                    min_dist = xbest,
                    assign_to_initial_centers_only = assign_to_initial_centers_only,
                    similarity_aggregation = similarity_aggregation,
                    folded_to_BitVect = folded_to_BitVect,
                    morgan_radius = morgan_radius,
                    morgan_NBits = morgan_NBits,
                    seed = seed)
        elif (initial_clustering_method == "iterative min overlap"):
            cls = iterative_clustering_by_minimal_overlap(list_of_molecules = None,
                   list_of_fingerprints = fps,
                   priority_to_removal_of_bits_from_S0 = priority_to_removal_of_bits_from_S0,
                   min_overlap = min_overlap,
                   min_sim = xbest,
                   folded_to_BitVect = folded_to_BitVect,
                   morgan_radius = morgan_radius,
                   morgan_NBits = morgan_NBits)[0]
        else:
            raise ValueError("initial_clustering_method can only be either 'sphere exclusion' or 'iterative min overlap'.")
    else:
        cls = tried_clusterings_dict[xbest]

    # Store the initial cluster numbers in df
    df[initial_cluster_column_name] = cls
    
    # And count how many clusters were made
    Nc = len(np.unique(cls))
        
    # If required, calculate the min inter-subsets distances

    if (calculate_min_inter_cluster_TanimotoDists == True):

        print("Calculating inter-cluster similarities (this may take a long time)...")
        # print("")        
        
        minisdists = min_intercluster_distances(list(df[initial_cluster_column_name]), fps)
        df[min_intercluster_distance_column_name] = minisdists
        
        # Calculate the median of the minimal inter-subset distances, per cluster
        
        median_dist_per_cluster = list(df[[min_intercluster_distance_column_name,initial_cluster_column_name]].groupby([initial_cluster_column_name]).median()[min_intercluster_distance_column_name])
        min_median_dist_per_cluster = min(median_dist_per_cluster)
    else:
        median_dist_per_cluster = []
        min_median_dist_per_cluster = 0

    print("Saving main output csv file in ",path_to_output_csv," ...")
    print("")
    # The use of sparse DataFrame columns causes horrible slowness in to_csv.
    # To counteract that, the file is written out in chunks, using an appropriately defined function.
    #df.to_csv(path_to_output_csv, index = False)
    write_csv_from_dataframe_with_sparse_cols(
        dataframe = df,
        sparse_columns_names = cols_with_data,
        output_csv_file_full_path = path_to_output_csv)
    print("Done.")
    print("")

# Function to use when data distributions don't matter: simple iterative split
# New version reading and writing file (rather than relying on provided mols or fps)

def iterative_split_by_minimal_overlap_from_csv(
    path_to_input_csv,
    path_to_output_csv,
    sizes,
    smiles_column_name = "SMILES",
    ML_subset_column_name = "ML_subset",
    priority_to_removal_of_bits_from_S0 = True,
    folded_to_BitVect = False,
    morgan_radius = 3,
    morgan_NBits = 32768,
    calculate_min_inter_subset_TanimotoDists = False,
    min_is_dist_column_name = "min_inter_subset_Tanimoto_distance"):
    """
    Takes as input a csv file with SMILES and any other columns.
    Splits the records into final ML subsets by chemistry, using the 'iterative minimal overlap' method.
    NOTE: this function is meant to be used when data distributions don't matter, and only a strongly chemistry-oriented split is required.
    
    Parameters
    ----------
    path_to_input_csv : string
        path to *pivoted* ((g)zipped) csv file with SMILES and any other columns.
    path_to_output_csv : string
        path to *pivoted* ((g)zipped) csv output file
        - this will be the input file with additional column ML_subset_column_name and, if required, min_is_dist_column_name
    sizes : list of positive numbers
        list of the desired final sizes (will be normalised to fractions internally)
        - NOTE: for speed, it is advised to sort the sizes ascending, e.g. [10, 10, 80], not [80, 10, 10]
        - However, the result depends on the order, so you may decide to put the largest size first.
    smiles_column_name : string
        name of the column containing the SMILES in the input csv
    ML_subset_column_name : string
        the name of the column, in the output csv, containing the index of the final ML subset each record was assigned to.
        NOTE: if sizes has length n, the values of ML_subset_column_name are 1 .. n
    priority_to_removal_of_bits_from_S0 : bool
        - see documentation of function 'iterative_clustering_by_minimal_overlap' for details
    folded_to_BitVect : bool
        must the fingerprints be calculated as folded bit vectors?
        - if True, the calculations will be faster but the results less accurate
        - if False, full fingerprints will be used, yielding more accurate results
    morgan_radius : integer {1, 2, 3}
        the radius for the morgan fingerprint calculation, where applicable
    morgan_NBits : positive integer
        the number of bits for the folded fingerprints (only applicable when folded_to_BitVect == True)
    calculate_min_inter_subset_TanimotoDists : bool
        must inter-subset distances be measured?
        - NOTE: this calculation requires a very long time, so do it only if strictly necessary
    min_is_distance_column_name : string        
        the minimal Tanimoto distance of the compound to any compound in any other subsets (if calculate_min_inter_subset_TanimotoDists == True).

    Returns
    -------
    List with:
    0. the subset each original record belongs to (= the content of ML_subset_column_name)
    1. the order of selection
    2. the current overlap
    3. the current number of bits in S0
    4. the current number of bits in S1
    5. the current size of S0
    6. the current size of S1
    7. the current Shannon entropy of S0
    8. the current Shannon entropy of S1
    9. the current Shannon entropy change of S0
    10. the current Shannon entropy change of S1    
    
    Saves:
    - a final output csv file with the same data as the input file, plus columns with:
      - the identifier of the final ML subset 
      - optionally, the inter-subset distances
      File name : path_to_output_csv
    """
    # START PROCESSING
        
    # Begin
    
    print("Split by minimal overlap")
    print("========================")
    print("")
    
    # 1. Read data in csv format (file with smiles)
    print("Reading csv file...")
    print("")
    # debug: this made a very large df, which can instead be made sparse
    #df = pd.read_csv(path_to_input_csv)
    # Use instead a function that reads columns in chunks and makes a dataframe with sparse columns

    df = read_csv_to_dataframe_with_sparse_cols(
        dense_columns_names = [smiles_column_name],
        input_csv_file_full_path = path_to_input_csv)
    
    #cols_with_data = df.columns.values[~(df.columns.isin([smiles_column_name]))]
    # small modification made on 20220127:
    # ensure that cols_with_data does not include any of the dense columns that are created by this process
    # (which might happen if the input file wrongly contains columns with those names, that are interpreted as data)
    cols_with_data = df.columns.values[~(df.columns.isin([smiles_column_name, ML_subset_column_name, min_is_dist_column_name]))]
    
    # 2,3. Convert the list of smiles to a list of fingerprints without storing molecules.
    print("Creating fingerprints from SMILES...")
    print("")

    if (folded_to_BitVect == True):
        case = "BitVect"
    else:
        case = "IntVect"
    # New in v20.2 : the fingerprints made by the new Morgan generator are bit vectors for both the folded and unfolded FP's
    case = "BitVect"

    fps = []

    # New in v20.2 : we use a Morgan FP generator, which must be initialised here
    if (folded_to_BitVect == True) :
        mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius = morgan_radius, fpSize = morgan_NBits)
    else :
        mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius = morgan_radius)
        # Not necessary to get the Bit Info Mapping, if we assume that GetOnBits() is consistent for both types of FP.
        #ao = rdFingerprintGenerator.AdditionalOutput()
        #ao.AllocateBitInfoMap()

    for s in df[smiles_column_name]:

        m = Chem.MolFromSmiles(s)

        if (folded_to_BitVect == True):
            #fps.append(rdMolDescriptors.GetMorganFingerprintAsBitVect(m, morgan_radius, morgan_NBits))
            fps.append(mfpgen.GetFingerprint(m))
        else:
            #fps.append(Chem.rdMolDescriptors.GetMorganFingerprint(m, morgan_radius, useCounts = False))
            fps.append(mfpgen.GetSparseFingerprint(m))
   
    # 4. Make fingerprint summaries as needed for the selection.
    
    print("Preparing fingerprint summaries...")
    print("")

    # 4.1. Create a dictionary with the global counts fingerprint 'fp0' for the initial set S0 with all molecules.
    # At the same time create a dictionary 'id_vs_fpbit' of which molecules contain which fpbits.
    # Also, make a dictionary of the list of bits of each compound, for easier reference.
    
    fp0 = dict()
    id_vs_fpbit = dict()
    fp_vs_id = dict()
    
    for i, fpi in enumerate(fps):
        
        if (case == "BitVect"):
            fpi_bits_list = [fpi.GetOnBits()[j] for j in range(fpi.GetNumOnBits())]
        else:
            fpi_bits_list = list(fpi.GetNonzeroElements().keys())            
        
        fp_vs_id[i] = fpi_bits_list
        
        for fpbit in fpi_bits_list:
            if fpbit in fp0:
                fp0[fpbit] += 1
            else:
                fp0[fpbit] = 1
            if fpbit in id_vs_fpbit:
                id_vs_fpbit[fpbit].append(i)
            else:
                id_vs_fpbit[fpbit] = [i]
    
    # 4.1.2. Make a frequencies dictionary, to speed up the calculation of Shannon entropy
    
    fp0_freqs = dict()
    
    for f in list(fp0.values()):
        if f in fp0_freqs:
            fp0_freqs[f] += 1
        else:
            fp0_freqs[f] = 1
    
    # Function definition no longer necessary thanks to the new SE calculation method
    #def Fr0(f):
    #    if f in fp0_freqs:
    #        Fr = fp0_freqs[f]
    #    else:
    #        Fr = 0
    #    return Fr
    #def Fr1(f):
    #    if f in fp1_freqs:
    #        Fr = fp1_freqs[f]
    #    else:
    #        Fr = 0
    #    return Fr
    
    # 4.1.3. Make a U(f) list for all required values of f (from 2 to len(fps)), so it's done only once
    # U(f) = f*log(f)
    
    Uf = [0] * (len(fps)+1)
    
    for f in range(2,len(fps)+1):
        Uf[f] = f * np.log2(f)
    
    # 4.2. Create a dictionary 'fp01' of fingerprint bits with value 1 in fp0 (i.e. those that would disappear if a compound
    # containing them were moved from S0 to S1).

    fp01 = {fpbit for fpbit,value in fp0.items() if value == 1}
    fp01 = dict.fromkeys(fp01, 0)
    
    # 4.3. For each compound, calculate how many bits it would remove from S0 if it were moved to S1, and store its negated value.
    # And calculate how many new bits it would add to S1 if it were moved to S1.
    # At this stage, S1 is empty, so this is simply the total number of bits in the compound.

    deltaS0 = []
    deltaS1 = []
    for i in fp_vs_id:
        deltaS0.append(0)
        for fpbit in fp_vs_id[i]:
            if fpbit in fp01:
                deltaS0[i] -= 1
        deltaS1.append(len(fp_vs_id[i]))
        
    # The delta overlap is the element-wise sum of the elements of deltaS0 and deltaS1

    delta_overlap = []
    for (dS0, dS1) in zip(deltaS0, deltaS1):
        delta_overlap.append(dS0+dS1)

    # 5. Make the partitions
    
    # For a single bipartition:

    # Initialise a dictionary S1, which will contain the indices of the compounds in S1, and a fingerprint fp1, for the bits in S1.
    # At each step, identify which compound corresponds to the minimal delta_overlap, resolving ties as specified by the user, and:
    # - using the fingerprint bits in the moved compound, update fp1, fp0, fp01, deltaS0, deltaS1 and delta_overlap
    # - loop, until a condition is met (e.g. the desired number of compounds are in S1)
    # - append the compound index to S1
    # - make sure that the delta_overlap of the selected item is set to a very high value, so it's not chosen

    # Once a bipartition is complete, reinitialise all the required dictionaries and start again, except for the last subset, which
    # simply follows by exclusion.
    
    # 5.1. First, create the correct numerical subset sizes from the user-defined 'sizes' list
    # Do NOT sort: while it is more efficient to make the small subsets first, the user must decide
    # In any case, normalise to sum 1
    
    fractional_sizes = sizes / np.sum(sizes)

    Nk = len(fps)
    S = len(sizes)

    # If the sizes are not all the same, use the ceiling and then adjust the largest (last) size if needed
    
    if S == 1:
        S = 2
        absolute_sizes = [Nk-1, 1]
    else:    
        if (len(np.unique(fractional_sizes)) != 1):
            absolute_sizes = [np.ceil(Nk*x).astype('int64') for x in fractional_sizes]
            absolute_sizes[len(absolute_sizes)-1] = Nk - sum(absolute_sizes[:(len(absolute_sizes)-1)])
        # If the sizes are all the same, distribute as equally as possible by specific maths
        else:
            Nbig = (np.ceil(Nk/S)).astype('int64')
            Nsmall = (np.floor(Nk/S)).astype('int64')
            n = (Nbig * S - Nk).astype('int64')
            absolute_sizes = []
            for i in range(n):
                absolute_sizes.append(Nsmall)
            for i in range(S-n):
                absolute_sizes.append(Nbig)
    
    # 5.2. Run the iterations

    clusters = defaultdict(list)
    order_of_selection_list = []
    current_overlap_list = []
    current_N_bits_S0_list = []
    current_N_bits_S1_list = []
    current_size_S0_list = []
    current_size_S1_list = []
    current_ShannonEntropy_S0_list = []
    current_ShannonEntropy_S1_list = []
    current_ShannonEntropy_S0_delta_list = []
    current_ShannonEntropy_S1_delta_list = []
    set_n = 0
    already_assigned = dict()
    order_of_sel = 0
    Nk_S0 = Nk
        
    for size in absolute_sizes[:(len(absolute_sizes)-1)]:

        set_n += 1
        S1 = dict()
        Nk_S1 = 0
        fp1 = dict()
        fp1_freqs = dict()
        total_overlap = [0]
        selection_iteration = 0
        num_bits_in_S0 = len(fp0)
        num_bits_in_S1 = len(fp1)
        num_bits_total = num_bits_in_S0
        current_overlap = num_bits_in_S0 + num_bits_in_S1 - num_bits_total
        # Calculation of of SE from bit frequencies (slow, only for initial testing)
        #SE_S0_from_bits = -1 * (sum([(fp0[b]/Nk_S0)*np.log2(fp0[b]/Nk_S0) for b in fp0 if fp0[b] > 0]) + \
        #              sum([(1-fp0[b]/Nk_S0)*np.log2(1-fp0[b]/Nk_S0) for b in fp0 if fp0[b] < Nk_S0]))
        SE_S1 = 0
        # SE calculated by the frequencies list method (faster)
        if (Nk_S0 > 0):
        #    SE_S0 = num_bits_in_S0 * np.log2(Nk_S0) - \
        #    (Fr0(Nk_S0) * Uf[Nk_S0] + 
        #     sum((Fr0(f)+Fr0(Nk_S0-f))*Uf[f] for f in set(list(fp0_freqs.keys())+[Nk_S0-f for f in fp0_freqs.keys()]) if ((f >= 2) & (f <= Nk_S0-1))) ) \
        #    / Nk_S0
        # New version, does not require the complicated set detection or Fr definition
            SE_S0 = num_bits_in_S0 * np.log2(Nk_S0) -             sum((Uf[f]+Uf[Nk_S0-f])*Fr for f,Fr in fp0_freqs.items())             / Nk_S0
        else:
            SE_S0 = 0
        
        print("Starting selection # ",set_n," / ",(len(absolute_sizes)-1)," (",size," compounds) ...")
        print("Initial situation:")
        print("num_bits_in_S0 = ", num_bits_in_S0)
        print("num_bits_in_S1 = ", num_bits_in_S1)
        print("current overlap = ", current_overlap)
        #print("SE_S0 from freqs = ", SE_S0)
        #print("SE_S0 from bits = ", SE_S0_from_bits)
                
        while len(S1) < size :
            
            order_of_sel += 1
            order_of_selection_list.append(order_of_sel)

            # The compound to move from S0 to S1 must have the lowest delta_overlap

            # sel = delta_overlap.index(min(delta_overlap))
            # Note: no longer based on delta_overlap alone; now ties are broken by deltaS, see below

            # Identify first the indices of all compounds with minimal delta_overlap
            min_dov = min(delta_overlap)
            min_dov_indices = [i for i, dov in enumerate(delta_overlap) if dov == min_dov]
            # Then find which has the lowest deltaS0 (if priority_to_removal_of_bits_from_S0 == True)
            # or the lowest deltaS1 (if priority_to_removal_of_bits_from_S0 == False)
            if (priority_to_removal_of_bits_from_S0 == True):
                deltaS = [deltaS0[i] for i in min_dov_indices]
            else:
                deltaS = [deltaS1[i] for i in min_dov_indices]
            # Select the compound to move
            sel = min_dov_indices[deltaS.index(min(deltaS))]
            
            # Process the fingerprint of the selected compound
            current_overlap_from_delta = total_overlap[selection_iteration]+delta_overlap[sel]
            total_overlap.append(current_overlap_from_delta)

            selection_iteration += 1
            Nk_S0 -= 1
            Nk_S1 += 1
            
            for fpbit in fp_vs_id[sel]:

                # Add the bits in the selected compound to fp1
                if fpbit in fp1:
                    fp1[fpbit] += 1
                else:
                    fp1[fpbit] = 1
                    num_bits_in_S1 += 1
                    # If the bit is new to fp1, the deltaS1 of all (still selectable) compounds containing this bit must decrease by 1
                    for x in id_vs_fpbit[fpbit]:
                        if x not in already_assigned:
                            deltaS1[x] -= 1
                            delta_overlap[x] -= 1
                
                # Adding this bit to fp1 increases by 1 the count of its new (current) frequency
                if fp1[fpbit] in fp1_freqs:
                    fp1_freqs[fp1[fpbit]] += 1
                else:
                    fp1_freqs[fp1[fpbit]] = 1
                # And it decreases by 1 the count of its previous frequency, except if it is 0
                if fp1[fpbit] > 1:
                    fp1_freqs[fp1[fpbit]-1] -= 1
                    if fp1_freqs[fp1[fpbit]-1] == 0:
                        fp1_freqs.pop(fp1[fpbit]-1)
                
                # Remove the bits in the selected compound from fp0
                fp0[fpbit] -= 1
                if (fp0[fpbit] == 1):
                    fp01[fpbit] = 0
                    # If the bit reaches count 1 in fp0, the deltaS0 of all (still selectable) compounds containing this bit must decrease by 1
                    for x in id_vs_fpbit[fpbit]:
                        if x not in already_assigned:
                            deltaS0[x] -= 1
                            delta_overlap[x] -= 1
                elif (fp0[fpbit] == 0):
                    fp01.pop(fpbit)
                    num_bits_in_S0 -= 1
                    # If the bit reaches count 0 in fp0, the deltaS0 of all (still selectable) compounds containing this bit must increase by 1
                    for x in id_vs_fpbit[fpbit]:
                        if x not in already_assigned:
                            deltaS0[x] += 1
                            delta_overlap[x] += 1
                
                # Removing this bit from fp0 decreases by 1 the count of its previous frequency
                fp0_freqs[fp0[fpbit]+1] -= 1
                if fp0_freqs[fp0[fpbit]+1] == 0:
                    fp0_freqs.pop(fp0[fpbit]+1)
                # And it increases by 1 the count of its new (current) frequency, except if it is 0
                if fp0[fpbit] > 0:
                    if fp0[fpbit] in fp0_freqs:
                        fp0_freqs[fp0[fpbit]] += 1
                    else:
                        fp0_freqs[fp0[fpbit]] = 1

                # Note: in previous implementations, the bits that reached frequency 0 in fp0 were not removed (.pop).
                #       However, not doing that forced one to use conditions like fp0[b] > 0 to avoid errors.
                #       It seems better to remove the bits from fp0. This also ensures len(fp0) = # bits in S0.
                if (fp0[fpbit] == 0):
                    fp0.pop(fpbit)
                                        
            current_overlap = num_bits_in_S0 + num_bits_in_S1 - num_bits_total
            
            # Calculate the current Shannon entropies            
            old_SE_S0 = SE_S0
            old_SE_S1 = SE_S1
            # Calculation of of SE from bit frequencies (slow, only for initial testing)
            #SE_S0 = -1 * (sum([(fp0[b]/Nk_S0)*np.log2(fp0[b]/Nk_S0) for b in fp0 if fp0[b] > 0]) + \
            #              sum([(1-fp0[b]/Nk_S0)*np.log2(1-fp0[b]/Nk_S0) for b in fp0 if fp0[b] < Nk_S0]))
            #SE_S1 = -1 * (sum([(fp1[b]/Nk_S1)*np.log2(fp1[b]/Nk_S1) for b in fp1 if fp1[b] > 0]) + \
            #              sum([(1-fp1[b]/Nk_S1)*np.log2(1-fp1[b]/Nk_S1) for b in fp1 if fp1[b] < Nk_S1]))
            # SE calculated by the frequencies list method (faster)
            if (Nk_S0 > 0):
            #SE_S0 = num_bits_in_S0 * np.log2(Nk_S0) - \
            #(Fr0(Nk_S0) * Uf[Nk_S0] + 
            # sum((Fr0(f)+Fr0(Nk_S0-f))*Uf[f] for f in set(list(fp0_freqs.keys())+[Nk_S0-f for f in fp0_freqs.keys()]) if ((f >= 2) & (f <= Nk_S0-1))) ) \
            #/ Nk_S0
                # New version, does not require the complicated set detection or Fr definition
                SE_S0 = num_bits_in_S0 * np.log2(Nk_S0) -                 sum((Uf[f]+Uf[Nk_S0-f])*Fr for f,Fr in fp0_freqs.items())                 / Nk_S0
            else:
                SE_S0 = 0
            
            #SE_S1 = num_bits_in_S1 * np.log2(Nk_S1) - \
            #(Fr1(Nk_S1) * Uf[Nk_S1] + 
            # sum((Fr1(f)+Fr1(Nk_S1-f))*Uf[f] for f in set(list(fp1_freqs.keys())+[Nk_S1-f for f in fp1_freqs.keys()]) if ((f >= 2) & (f <= Nk_S1-1))) ) \
            #/ Nk_S1
            # New version, does not require the complicated set detection or Fr definition
            SE_S1 = num_bits_in_S1 * np.log2(Nk_S1) -             sum((Uf[f]+Uf[Nk_S1-f])*Fr for f,Fr in fp1_freqs.items())             / Nk_S1
            
            current_ShannonEntropy_S0_list.append(SE_S0)
            current_ShannonEntropy_S1_list.append(SE_S1)
            delta_SE_S0 = (SE_S0 - old_SE_S0)
            delta_SE_S1 = (SE_S1 - old_SE_S1)
            current_ShannonEntropy_S0_delta_list.append(delta_SE_S0)
            current_ShannonEntropy_S1_delta_list.append(delta_SE_S1)            
            
            # Update the metrics lists
            current_overlap_list.append(current_overlap)
            current_N_bits_S0_list.append(num_bits_in_S0)
            current_N_bits_S1_list.append(num_bits_in_S1)
            current_size_S0_list.append(Nk_S0)
            current_size_S1_list.append(Nk_S1)
                        
            #print("Selected ",selection_iteration,"/",size)
            #print("num_bits_in_S0 = ", num_bits_in_S0)
            #print("num_bits_in_S1 = ", num_bits_in_S1)
            #print("current overlap = ", current_overlap)
            #print("current overlap from delta = ", current_overlap_from_delta)
            #print("SE_S0 from bit fractions = ", SE_S0)
            #print("SE_S0 from freqs = ", SE_S0_from_freqs)
            #print("SE_S1 from bit fractions = ", SE_S1)
            #print("SE_S1 from freqs = ", SE_S1_from_freqs)

            # Append the selected compound to S1 and to already_assigned
            S1[sel] = 1
            already_assigned[sel] = set_n
            # Set the delta_overlap of sel to a very high value
            delta_overlap[sel] = len(fp0)

        print("Done. Final situation:")
        print("num_bits_in_S0 = ", num_bits_in_S0)
        print("num_bits_in_S1 = ", num_bits_in_S1)
        print("current overlap = ", current_overlap)
        # plt.plot(total_overlap); plt.xlabel('iteration'); plt.ylabel('overlap')
        print("")

        # Append the result of this selection to the main clusters dictionary
        for i in S1:
            clusters[set_n].append(i)
        # Reinitialise, of course only if there is another selection to make
        if (set_n < len(absolute_sizes)-1):
            fp00 = [fpbit for fpbit,value in fp0.items() if value == 0]
            for fpbit in fp00:
                fp0.pop(fpbit)
                # id_vs_fpbit.pop(fpbit) # not really necessary; the bits that are no longer in fp0 will not be searched
            # All compounds that are still selectable and whose bits will disappear from S1 when S1 is emptied need to have their
            # deltaS1 and delta_overlap corrected as appropriate.    
            for fpbit in fp1:
                for x in id_vs_fpbit[fpbit]:
                    if x not in already_assigned:
                        deltaS1[x] += 1
                        delta_overlap[x] += 1
            # After which, fp1 and S1 can be safely reinitialised (which happens at the beginning of the loop).

    # Finally, fill the last subset
    for sel in range(Nk):
        if sel not in already_assigned:
            clusters[S].append(sel)
            order_of_selection_list.append(order_of_sel+1)
            current_overlap_list.append(current_overlap)
            current_N_bits_S0_list.append(num_bits_in_S0)
            current_N_bits_S1_list.append(num_bits_in_S1)
            current_size_S0_list.append(0)
            current_size_S1_list.append(Nk_S0)
            current_ShannonEntropy_S0_list.append(0)
            current_ShannonEntropy_S1_list.append(SE_S0)
            current_ShannonEntropy_S0_delta_list.append(0)
            current_ShannonEntropy_S1_delta_list.append(0)
    
    # END: output
    
    # Create the permutation order mapping the input set to the order of selection
    indices = []
    for c in clusters:
        indices.extend(clusters[c])
    
    order = [x for _,x in sorted(zip(indices, range(Nk)))]
    
    # Map the permutation on all relevant lists
    
    cls = np.repeat([i+1 for i in range(S)], absolute_sizes)
    cls = [cls[i] for i in order]
    order_of_selection_list = [order_of_selection_list[i] for i in order]
    current_overlap_list = [current_overlap_list[i] for i in order]
    current_N_bits_S0_list = [current_N_bits_S0_list[i] for i in order]
    current_N_bits_S1_list = [current_N_bits_S1_list[i] for i in order]
    current_size_S0_list = [current_size_S0_list[i] for i in order]
    current_size_S1_list = [current_size_S1_list[i] for i in order]
    current_ShannonEntropy_S0_list = [current_ShannonEntropy_S0_list[i] for i in order]
    current_ShannonEntropy_S1_list = [current_ShannonEntropy_S1_list[i] for i in order]
    current_ShannonEntropy_S0_delta_list = [current_ShannonEntropy_S0_delta_list[i] for i in order]
    current_ShannonEntropy_S1_delta_list = [current_ShannonEntropy_S1_delta_list[i] for i in order]

    df[ML_subset_column_name] = cls
    
    # If required, calculate the min inter-subsets distances

    if (calculate_min_inter_subset_TanimotoDists == True):

        print("Calculating inter-ML-subsets similarities (this may take a long time)...")
        # print("")        
        
        minisdists = min_intercluster_distances(cls, fps)
        df[min_is_dist_column_name] = minisdists        
    
    # Save output file
    
    print("Saving main output csv file in ",path_to_output_csv," ...")
    print("")
    # The use of sparse DataFrame columns causes horrible slowness in to_csv.
    # To counteract that, the file is written out in chunks, using an appropriately defined function.
    #df.to_csv(path_to_output_csv, index = False)
    write_csv_from_dataframe_with_sparse_cols(
        dataframe = df,
        sparse_columns_names = cols_with_data,
        output_csv_file_full_path = path_to_output_csv)
    
    # Return the results
    
    return [cls,
            order_of_selection_list,
            current_overlap_list,
            current_N_bits_S0_list,
            current_N_bits_S1_list,
            current_size_S0_list,
            current_size_S1_list,
            current_ShannonEntropy_S0_list,
            current_ShannonEntropy_S1_list,
            current_ShannonEntropy_S0_delta_list,
            current_ShannonEntropy_S1_delta_list]

