# v18.2 :
# - F1_SDF_to_csv_with_SMILES : added option to store the CTAB '__Name' property into a data property in the output 
# - F4_csv_unpivoted_std_transf_cur_to_averaged : added option to remove outliers when calculating the mean


# import required packages

import pandas as pd
import numpy as np
import rdkit
from rdkit import Chem
from rdkit.Chem import rdmolops
from chembl_structure_pipeline import standardizer
from chembl_structure_pipeline import checker
from rdkit.Chem.MolStandardize import rdMolStandardize
largest_Fragment = rdMolStandardize.LargestFragmentChooser()
import math
from math import exp, log10, log
import gzip
import csv
import codecs
utf8 = codecs.getwriter('utf_8')

# this is necessary to disable errors and warnings from rdkit, which seem to slow down the process and occupy too much space
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# this is necessary to disable errors and warnings from numpy, for clearer output by the data analysis function
np.seterr(all="ignore")

# define functions

# Function that returns the chirality flag '0' or '1' when the molblock is V2000, otherwise '1'
def chiral_flag_from_molblock(molblock) :
    """
    Parse a molblock (CTAB) as text to find the chiral flag ('0' or '1'), if V2000.
    If V3000, return '-1'.
    
    Parameters
    ----------
    molblock : str
        The molblock to process. A structured string with '\n' to delimit lines.
    
    Returns
    -------
    str
        The chiral flag, '0' or '1'.
    """
    # *** VERY IMPORTANT NOTE 2024-03-06 ***
    # It was discovered that some improperly formatted CTAB's make this mechanism ineffective.
    # Function CTAB_to_CXSMILES modified so it does not depend on this function anymore.
    # See source of better chirality flag detection mechanism : https://gist.github.com/greglandrum/f85097a8489ba4a5825b0981b1fd2408
    # The counts line is supposed to be the 4th row in the V2000 CTAB
    counts_line = molblock.split('\n')[3]
    cf = '-1'
    if 'V2000' in counts_line :
        cl_split = counts_line.split(' ')
        # The chirality flag is supposed to be the 5th non-empty element in the counts line
        i = 0
        for d in cl_split :
            if d != '' :
                i += 1
            if i == 5 :
                cf = d
                break
    return cf

# Function that converts V2000 and V3000 CTAB's (molblocks) to CXSMILES
def CTAB_to_CXSMILES(CTAB) :
    """
    Convert a V2000 or V3000 CTAB to CXSMILES.

    - V2000 CTAB's can only encode stereochemically absolute or racemic compounds,
      depending on the 'chiral flag' (a 0 or 1 in a specific place of the CTAB text).
    - rdkit intentionally ignores the chiral flag when generating a CXSMILES from
      a molecule made from a V2000 CTAB (confirmed by Greg Landrum).
    - This function handles the issue by parsing the chiral flag, making a plain SMILES
      and attaching either no enhanced stereo string when chiral flag is '1', or
      attaching the appropriate '&1' enhanced stereo string when chiral flag is '0'.
    - V3000 CTAB's can encode more complex ('enhanced') stereochemistry compared to
      V2000, e.g. molecules with an absolute stereocenter and a racemic one
      (i.e. a diastereoisomeric mixture).
    - For molecules made from V3000 CTAB's, the standard function Chem.MolToCXSmiles()
      generates valid CXSMILES, except that it adds too much information (coordinates...).
    - --> found how to limit the output to the enhanced string only.
    - added in v15.1 : remove any possibly existing molblock atom numbering from the molecule
    IMPORTANT NOTE: the core logic of this function has no error catch mechanism, so it
      *will* err out and halt when it generates an error.
      Ensure to catch errors by wrapping its call as appropriate.
    ALSO NOTE: empty CTAB's and any CTAB's that fail to be converted to a non-None molecule
      do NOT generate an error, but an '' output. Ensure to catch that
      case a posteriori if you want to always have an actual non-empty CXSMILES.
    
    Parameters
    ----------
    CTAB : str
        The CTAB to process.
    
    Returns
    -------
    str
        The CXSMILES.
    """
    mol = None
    try :
        mol = Chem.MolFromMolBlock(CTAB)
    except :
        pass
    
    if mol == None :
        return ''
    
    # Remove atom mapping info, in case it is present in the input molblock (was the case for a few Lims mols...)
    for a in mol.GetAtoms() :
        a.SetAtomMapNum(0)    

    # By default, assume that the CTAB is V3000 (cf = -1)
    cf = '-1'
    # Check if instead this is a V2000 CTAB
    counts_line = CTAB.split('\n')[3]
    if 'V2000' in counts_line :
        # If so, assume the chirality flag is 1 (ABS)
        cf = '1'
        # Then detect the chirality flag as per https://gist.github.com/greglandrum/f85097a8489ba4a5825b0981b1fd2408
        if mol.HasProp('_MolFileChiralFlag'):
            cf = str(mol.GetIntProp('_MolFileChiralFlag'))

    # Only if the CTAB is V2000 and chirality flag is 0 (RAC), apply the method that adds all stereocenters to a single 'and' group
    # See https://github.com/rdkit/rdkit/issues/7557
    if cf == '0' :
        #print("RAC chirality flag - looking for sp3 stereocenters and atropoisomeric axes...")
        # Are there any tetrahedral sp3 stereocenters?
        aids = [at.GetIdx() for at in mol.GetAtoms() if at.GetChiralTag() in \
                (Chem.ChiralType.CHI_TETRAHEDRAL_CCW, Chem.ChiralType.CHI_TETRAHEDRAL_CW)]
        # Are there any atropoisomeric axes?
        atropo_aids = [bo.GetBeginAtomIdx() for bo in mol.GetBonds() if bo.GetStereo() in \
                       (Chem.rdchem.BondStereo.STEREOATROPCW, Chem.rdchem.BondStereo.STEREOATROPCCW) ]
        atropo_aids2 = [bo.GetEndAtomIdx() for bo in mol.GetBonds() if bo.GetStereo() in \
                        (Chem.rdchem.BondStereo.STEREOATROPCW, Chem.rdchem.BondStereo.STEREOATROPCCW) ]
        aids.extend(atropo_aids)
        aids.extend(atropo_aids2)
        # If any of the above were found, make a single 'and' stereogroup and include the relevant atoms into it
        if aids :
            #print(f"Found {str(len(aids))}  atoms to set to 'and'.")
            sgt = Chem.StereoGroupType.STEREO_AND
            mol = Chem.RWMol(mol)
            sgs = list(mol.GetStereoGroups())
            ng = Chem.CreateStereoGroup(sgt, mol, aids)
            sgs.append(ng)
            mol.SetStereoGroups(sgs)
            mol = mol.GetMol()

    # New in v18: we now include CX_BOND_ATROPISOMER)
    smilesWriterParams = Chem.SmilesWriteParams()
    cxsm = Chem.MolToCXSmiles(mol, smilesWriterParams,
                              Chem.rdmolfiles.CXSmilesFields.CX_ENHANCEDSTEREO | Chem.CXSmilesFields.CX_BOND_ATROPISOMER)
    return cxsm

# Function that converts a SDF to csv with CXSMILES
def F1_SDF_to_csv_with_SMILES(
    input_SD_file_path,
    fail_output_SD_file_path,
    pass_output_csv_file_path,
    output_SMILES_colname = 'SMILES',
    put_name_into_data_property = None,
    ):
    """
    Transform a SD file into a csv file with CXSMILES for the structures.
    All other data possibly present in the CTAB's of the SD file are also saved to the csv.
    
    Parameters
    ----------
    input_SD_file_path : str
        - path to the SD file to read
        - .csv.zip or .csv.gz files can be read as such, no need to unzip them elsewhere
    fail_output_SD_file_pat : str
        - path to the SD file to write out any failed SD records
        - can end in .sdf, .sdf.gz, .sdf.zip        
    pass_output_csv_file_path : str
        - path to the csv file to write out correctly processed, valid records
        - can end in .csv, .csv.gz, .csv.zip    
    output_SMILES_colname : str
        - defaults to 'SMILES'
        - mandatory column name for the CXSMILES column in the pass_output_csv_file_path file
    put_name_into_data_property : str or None
        - defaults to None
        - if set to a string, the 'private property' '__Name' attached to each CTAB, i.e. the first line
          of the header block, is copied to a column named as the provided string;
          this may be useful when the '__Name' contains some identifier that is not present elsewhere in the SD file.
          NOTE: if the put_name_into_data_property column is already contained in the SD file as a data property, it is overwritten!
    
    Returns
    -------
    None
        Saving the result file(s) is the only expected action by this function.
    """
    
    ext = input_SD_file_path.split(".")[-1].lower()
    if (ext != 'sdf') & (ext != 'gz') & (ext != 'zip'):
        raise ValueError('The input SD file can only have extension .SDF, .SDF.gz or .SDF.zip')

    ext_fail = fail_output_SD_file_path.split(".")[-1].lower()    
    if (ext_fail != 'sdf') & (ext_fail != 'gz') & (ext_fail != 'zip') :
        raise ValueError('The failed records SD file can only have extension .SDF, .SDF.gz or .SDF.zip')

    ext_out = pass_output_csv_file_path.split(".")[-1].lower()
    if (ext_out != 'csv') & (ext_out != 'gz') & (ext_out != 'zip') :
        raise ValueError('The passed records csv file can only have extension .csv, .csv.gz or .csv.zip')
    
    print("")
    print("Reading file " + input_SD_file_path)
    print("")

    # To be able to output wrong records when reading an SDF, the SDF must be parsed as text.
    # Define first the SDF record readers

    def read_record(fh):
        lines = []
        for line in fh:
            lines.append(line)
            if line.rstrip() == '$$$$':
                return ''.join(lines)

    def read_records(fh):
        while True:
            rec = read_record(fh)
            if rec == None:
                return
            yield rec

    # Using Chem.SDMolSupplier to read molecules AND data upfront
            
    invalid_mols_indices = []
    valid_mols_indices = []    
    index = 0
    data_dict = []
    field_names = dict()

    print(" > if any records are unreadable as molecules, they will be saved to: " + str(fail_output_SD_file_path))
    print("")

    sup = Chem.SDMolSupplier()

    if ext_fail == 'sdf' :
        # unzipped SDF
        failed_SDF_recs = open(fail_output_SD_file_path, 'wt')
    elif ext_fail == 'gz' :
        # gzipped SDF
        failed_SDF_recs = gzip.open(fail_output_SD_file_path, 'wt')
    elif ext_fail == 'zip' :
        # Windows zipped SDF
        import zipfile
        import os
        if os.path.exists(fail_output_SD_file_path) :
            os.remove(fail_output_SD_file_path)
        failed_SDF_recs_zip = zipfile.ZipFile(fail_output_SD_file_path, mode = 'a')
        failed_SDF_recs = failed_SDF_recs_zip.open(fail_output_SD_file_path.split('/')[-1].replace('.zip',''), 'w')
        
    if ext == 'sdf':
        # unzipped SDF
        SDF = open(input_SD_file_path, mode = 'rt')
    elif ext == 'gz':
        # gzipped SDF            
        SDF = gzip.open(input_SD_file_path, mode = 'rt')
    elif ext == 'zip':
        # Windows zipped SDF
        import zipfile
        import io
        zipped = zipfile.ZipFile(input_SD_file_path)
        files = zipped.namelist()                
        # currently assuming that there is only 1 file in the zip archive
        # (or that only the first file matters)            
        SDF_binary = zipped.open(files[0])
        SDF = io.TextIOWrapper(SDF_binary)
    
    for rec in read_records(SDF):
        sup.SetData(rec)
        m = next(sup)
        if m != None :
            try:
                #s = Chem.MolToSmiles(m)
                # new: we now process the text records directly and make a CXSMILES, not a plain SMILES
                s = CTAB_to_CXSMILES(rec)
            except:
                s = None
        # New: added condition for s == '' (when CTAB is empty)
        if (m == None) | (s == None) | (s == ''):
            invalid_mols_indices.append(index)
            if ext_fail == 'zip':
                failed_SDF_recs.write(bytes(rec + '\n', encoding = 'utf-8'))
            else:
                failed_SDF_recs.write(rec)
        else:
            valid_mols_indices.append(index) 
            d = dict({output_SMILES_colname : s})
            # Add the '__Name' property if so requested
            if put_name_into_data_property != None :
                d[put_name_into_data_property] = str(m.GetProp('_Name'))
            di = m.GetPropsAsDict()
            # remove the SMILES from the SDF data, if existing
            di.pop(output_SMILES_colname, '')
            # remove the put_name_into_data_property from the SDF data, if existing
            if put_name_into_data_property != None :
                di.pop(put_name_into_data_property, '')
            # merge the SMILES from the SDF mol and the SDF data
            d = {**d, **di}
            for k in d.keys():
                if k not in field_names:
                    field_names[k] = 1            
            data_dict.append(d)
        index += 1

    print("   > # readable molecules found : " + str(len(valid_mols_indices)) + " / " + str(index))
    print("")

    # save data as csv
    print("   > # saving final pass data to : " + str(pass_output_csv_file_path))
    print("")

    if ext_out == 'csv' :
        # unzipped csv
        csvfile = open(pass_output_csv_file_path, 'wt')
        writer = csv.DictWriter(csvfile, fieldnames = list(field_names.keys()))
    elif ext_out == 'gz' :
        # gzipped csv
        csvfile = gzip.open(pass_output_csv_file_path, 'wt')
        writer = csv.DictWriter(csvfile, fieldnames = list(field_names.keys()))
    elif ext_out == 'zip' :
        # Windows zipped csv
        import zipfile
        import os
        if os.path.exists(pass_output_csv_file_path) :
            os.remove(pass_output_csv_file_path)
        zip = zipfile.ZipFile(pass_output_csv_file_path, mode = 'a')
        csvfile = zip.open(pass_output_csv_file_path.split('/')[-1].replace('.zip',''), 'w')
        writer = csv.DictWriter(utf8(csvfile), fieldnames = list(field_names.keys()))    
    
    writer.writeheader()
    writer.writerows(data_dict)

# Function that reads a csv to a sparse dataframe with all sparse columns, with much lower memory usage than pandas.read_csv for very sparse sets
def F2_AUXFUN_read_csv_to_dataframe_with_all_sparse_cols(
    input_csv_file_full_path,
    max_N_cells = 100000000,
    cols_to_read = None,
    ):
    """
    Reads a csv to a sparse pandas DataFrame, saving memory in reading and storing (especially if sparse).

    Parameters
    ----------
    input_csv_file_full_path : str
        - path to the csv file to read
        - NOTE: .zip or .gz csv files can be read as such, no need to unzip them elsewhere
    max_N_cells : integer
        - defaults to 100000000
        - the maximal number of cells (rows * columns) that can be read in each chunk
        - this will determine the peak memory used during reading
        - if you get memory errors during reading, reduce this parameter
    cols_to_read : None or list
        - defaults to None
        - if None, all the columns in the file will be read
        - if a list of strings, only the specified columns will be read

    Returns
    -------
        A pandas DataFrame, where all columns are of type pandas.arrays.SparseArray
    """
        
    # Find the total number of columns in the file, and their names
    firstrow = pd.read_csv(input_csv_file_full_path, nrows = 1, usecols = cols_to_read)
    colnames = list(firstrow.columns)
    N_cols = len(colnames)

    # Calculate the number of rows to read per iteration,
    # depending on the number of columns and max number of cells to load per iteration

    N_rows_to_read_per_iteration = np.ceil(max_N_cells / N_cols).astype('int32')

    # Initialise an empty DataFrame
    df = pd.DataFrame()

    # Iterate to fill the DataFrame
    it = 1
    with pd.read_csv(input_csv_file_full_path,
                chunksize = N_rows_to_read_per_iteration,
                na_values = '', keep_default_na = False,
                compression = 'infer', usecols = cols_to_read) as rdr:
        for data in rdr:
            print("Reading chunk # " + str(it))
            for i in range(N_cols):
                data[colnames[i]] = pd.arrays.SparseArray(data[colnames[i]])    
            df = pd.concat([df, data], axis = 0, ignore_index = True)
            it += 1

    return df

# Function that unpivots a dataframe, handling sparse columns that are not compatible with pandas.melt
def F2_AUXFUN_unpivot_dataframe(
    df_piv,
    ID_columns_names,
    data_columns_names = None,
    property_column_name = 'MODULE',
    value_column_name = 'VALUE',
    ):
    """
    Unpivots a dataframe that has pandas.arrays.SparseArray columns (for which pd.melt would fail).
    
    Parameters
    ----------
    df_piv
        - the input pd.DataFrame
    ID_columns_names : list of strings
        - the column(s) by which records are unmerged
        - use a list of column names, even if only one!        
    data_columns_names : None or list of strings
        - the columns in df_piv which will become values of the property column
        - defaults to None (when all non-ID columns are to be considered as data)
        - if not None, use a *list* of column names, even if only one column        
    property_column_name : str
        - defaults to 'MODULE'
        - the name of the column that will contain the previous data columns names    
    value_column_name : str
        - defaults to 'VALUE'
        - the name of the column that will contain the previous values from the data columns
        
    Returns
    -------
    pandas.DataFrame
        The unpivoted (dense) pandas.DataFrame resulting from the unpivoting operations.
    """

    df_piv.reset_index(drop = True, inplace = True)

    for c in ID_columns_names :
        # some sanity checks
        if c not in list(df_piv.columns) :
            raise ValueError(c + ' is not a column of your input file. It is mandatory to have all the ID columns in the input. Please review your data.')
        if any(df_piv[c].isna()) :
            raise ValueError(c + ' is not defined in at least one row of your input file. It is mandatory to have all the ID columns defined in all rows in the input. Please review your data.')
        # convert sparse to dense if applicable
        try:
            df_piv[c] = df_piv[c].sparse.to_dense()
        except:
            pass        

    if data_columns_names == None :
        data_columns_names = [m for m in df_piv.columns if m not in ID_columns_names]

    unpiv_indices = []
    df_unpiv = pd.DataFrame()

    for c in data_columns_names:
        c_data = df_piv[c]
        c_data_notna_bool = c_data.notna()
        c_indices = c_data.index[c_data_notna_bool]
        c_N_data = len(c_indices)
        if c_N_data > 0 :
            unpiv_indices.extend(c_indices)
            df_c = pd.DataFrame({property_column_name : [c] * c_N_data,
                                value_column_name : list(c_data[c_data_notna_bool])})
            df_unpiv = pd.concat([df_unpiv, df_c], axis = 0, ignore_index = True)

    df_unpiv = pd.concat([df_piv.loc[unpiv_indices, ID_columns_names].reset_index(drop = True), df_unpiv], axis = 1)

    return df_unpiv

# Function that unpivots a csv, handling sparse columns that are not compatible with pandas.melt, and with much lower memory usage if very sparse
def F2_csv_pivoted_to_unpivoted(
    input_pivoted_csv_file_path,
    output_unpivoted_csv_file_path,
    SMILES_colname = 'SMILES',
    original_ID_colname = 'ID',
    columns_to_remove = [],
    output_SMILES_colname = 'SMILES',
    output_ID_colname = 'ID',
    output_MODULE_colname = 'MODULE',
    output_VALUE_colname = 'VALUE',
    ):
    """
    Transforms a pivoted csv file (with CXSMILES as structures) into an unpivoted csv file.
    Optionally, columns that are irrelevant can be specified, and are skipped.
    
    Parameters
    ----------
    input_pivoted_csv_file_path : str
        - path to the pivoted csv file to read
        - .csv.zip or .csv.gz files can be read as such, no need to unzip them elsewhere    
    output_unpivoted_csv_file_path : str
        - path to the csv file to write out to
        - can end in .csv, .csv.gz, .csv.zip    
    SMILES_colname : str
        - defaults to 'SMILES'
        - mandatory column name for the CXSMILES column in the input file
        - must be defined in all rows in the file    
    original_ID_colnam : str
        - defaults to 'ID'
        - mandatory column name for the identifier of each compound in the input file
          > if the column exists in the input file, it must be defined in all rows
          > if the column does NOT exist in the input file, the SMILES_colname will be copied to it
        - this column is most often used for internal reference, esp. for corporate compounds    
    columns_to_remove : list of strings
        - defaults to []
        - list of columns in the input file that must NOT be retained in the unpivoted output,
          i.e. columns that are NOT data and must NOT become MODULE column values
    output_SMILES_colname : str
        - defaults to 'SMILES'
        - the name of the CXSMILES column in the output file
        - must be different from the MODULE and VALUE colnames
    output_ID_colname : str
        - defaults to 'ID'
        - the name of the original identifier column in the output file
        - must be different from the MODULE and VALUE colnames
    output_MODULE_colname : str
        - defaults to 'MODULE'
        - the name of the column that will contain the previous data columns names, in the output file
    output_VALUE_colname : str
        - defaults to 'VALUE'
        - the name of the column that will contain the previous values from the data columns    
    
    Returns
    -------
    None
        Saving the resulting unpivoted file is the only expected action by this function.
    """
    
    if original_ID_colname == '' :
        original_ID_colname = None
    
    columns_to_remove = [c for c in columns_to_remove if c not in [SMILES_colname, original_ID_colname]]

    print("")
    print("Reading file " + input_pivoted_csv_file_path)
    print("")

    df = F2_AUXFUN_read_csv_to_dataframe_with_all_sparse_cols(
            input_csv_file_full_path = input_pivoted_csv_file_path,
            max_N_cells = 100000000,
            cols_to_read = None)
    
    # Check that SMILES_colname exists and is defined in all rows
    if SMILES_colname not in list(df.columns) :
        raise ValueError(SMILES_colname + ' is not present in the input file. Please review your data.') 
    if any(df[SMILES_colname].isna()) :
        raise ValueError(SMILES_colname + ' is NA in at least one row. Please review your data.')

    # If the ID column is not defined, copy the SMILES one to it
    if original_ID_colname not in list(df.columns) :
        df[original_ID_colname] = df[SMILES_colname].copy()
    else:
        # Check that original_ID_colname is defined in all rows
        if any(df[original_ID_colname].isna()) :
            raise ValueError(original_ID_colname + ', if present, must be defined in all rows. Please review your data.')
        # Check that no different SMILES exist for single ID's
        N_unique_SMILES_per_ID = df.groupby(original_ID_colname)[SMILES_colname].nunique()
        IDs_non_unique = [str(ID) for ID in N_unique_SMILES_per_ID[N_unique_SMILES_per_ID != 1].index]
        if len(IDs_non_unique) > 0 :
            raise ValueError(original_ID_colname + ' = [' + ','.join(IDs_non_unique) + '] \
                each corresponds to more than one distinct ' + SMILES_colname + '. \
                This is a data integrity breach. Please review your data.')

    # Remove the columns the user indicated, if any
    df.drop(columns = columns_to_remove, errors = 'ignore', inplace = True)

    # Unpivot
    df2 = F2_AUXFUN_unpivot_dataframe(
        df_piv = df,
        ID_columns_names = [SMILES_colname, original_ID_colname], # list of strings, !even if only one! (must be defined and not NA in *all* rows)
        data_columns_names = None, # list of strings, !even if only one!; or set to None if you want all non-ID columns to be data        
        property_column_name = output_MODULE_colname, # string; must not be equal to any of the ID_columns_names
        value_column_name = output_VALUE_colname # string; must not be equal to any of the ID_columns_names
    )

    del df

    # Rename the ID and SMILES columns in the unpivoted dataframe
    df2.rename(columns = {original_ID_colname : output_ID_colname, SMILES_colname : output_SMILES_colname}, inplace = True, errors = 'ignore')

    # Save to file
    print("")
    print("Saving data to " + output_unpivoted_csv_file_path)
    print("")

    df2.to_csv(output_unpivoted_csv_file_path, index = False)


# Found by testing: a custom largest fragment picker function is needed, as the OOB rdkit one suppresses stereogroups

def largest_fragment_picker(CXSMILES,
                            verbose = False,
                           ) :
    """
    Takes a (CX)SMILES as input.
    Identifies the molecular fragment with the largest number of atoms.
    Returns the CXSMILES of that fragment.
      
    Parameters
    ----------
    CXSMILES : str
        The (CX)SMILES to process.
    verbose : bool
        Defaults to False. Set to True *only for very small sets of compounds* to investigate the output more granularly.
        
    Returns
    -------
    str
        The CXSMILES of the largest fragment found in the input CXSMILES.
        Or None when any of the parts of the CXSMILES was invalid.
    """
    
    # Temporarily assign the CXSMILES as output, in case no processing is needed
    cxsm_out = CXSMILES
    if verbose == True :
        print('Input CXSMILES = ' + str(CXSMILES))
    # Separate the enhanced stereo string, if any, from the SMILES
    split_1 = str(CXSMILES).split('|')
    # Isolate the SMILES
    all_SMILES = split_1[0]
    # Isolate the individual fragments in the SMILES
    split_2 = all_SMILES.split('.')
    # If there are more than one fragment in the SMILES, do the largest fragment picking
    if len(split_2) > 1 :        
        if verbose == True :
            print('Found ' + str(len(split_2)) + ' fragments.')
            print('Looking for largest one...')
        num_atoms_each = []
        max_num_atoms = 0
        index_picked = 0
        for i, sm in enumerate(split_2) :
            # Convert fragment to molecule
            # (errors can happen here if the sm is invalid; catch them externally)
            # (also, None is returned, stopping the process, if the fragment is not converted to a valid molecule)
            mol = Chem.MolFromSmiles(sm)
            if mol == None :
                if verbose == True :
                    print('Invalid fragment found. Stopping the process.')
                return None
            # Count the atoms in the fragment
            num_atoms = mol.GetNumAtoms()
            # Add it to the num_atoms_each list
            num_atoms_each.append(num_atoms)
            # If the fragment is the largest so far, store the info
            if num_atoms > max_num_atoms :
                index_picked = i
                max_num_atoms = num_atoms
        # The chosen fragment becomes the temporary output; strip spaces
        cxsm_out = split_2[index_picked].strip(' ')
        num_atoms_out = max_num_atoms
        if verbose == True :
            print('Largest fragment found = ' + cxsm_out + ' (' + str(num_atoms_out) + ' atoms).')
        # If there was any enhanced stereo, shift the indices as appropriate,
        # also eliminating any indices or groups from other fragments, if any
        if len(split_1) == 3 :
            shift = -1 * sum(n for n in num_atoms_each[:index_picked])
            # Enhanced stereo strings are expected to have format:
            # Grouptype0Groupnumber0:aidx00,aidx01,Grouptype1Groupnumber1:aidx10,aidx11,...                        
            if verbose == True :
                print('Enhanced stereo string = ' + split_1[1])
                print('Parsing individual groups and renumber atom indices...')
            split_3 = split_1[1].split(':')            
            enh_st_index = 0
            new_enhanced_stereo_string = ''
            while enh_st_index < len(split_3) - 1 :
                grouptypeandnumber = split_3[enh_st_index]
                if verbose == True :
                    print('Group starter found = ' + grouptypeandnumber)
                # if the first character of grouptypeand number is an integer instead of 'a' or '&' or 'o', there is a problem
                try:
                    first_element_of_group = int(grouptypeandnumber[0])
                    if verbose == True :
                        print('Invalid group starter. Stopping the process.')
                    return None
                except:
                    pass
                second_part = split_3[enh_st_index + 1]
                # second part may contain only atom indices, if there is only one group,
                # or the atom indices, plus the beginning of the next group, which is then the last element
                split_4 = second_part.split(',')
                if verbose == True :
                    print('> Elements of group found = ' + str(split_4))
                # if the last element of this list is not an integer, it's the next grouptypeandnumber
                try:
                    int(split_4[-1])
                except:
                    split_3[enh_st_index + 1] = split_4.pop()
                    if verbose == True :
                        print('> Last element of group was starter of next group. Reassigning.')
                # shift each index in split_4 (which is now the list of atom indices of the current group),
                # removing cases where the index falls out of the allowed range (i.e. the group was in a smaller fragment)
                if verbose == True :
                    print('> Renumbering atom indices in group and discarding invalid ones...')
                for j, enh_st_ij in enumerate(split_4) :
                    try:
                        aidx = int(enh_st_ij)
                        aidx_shifted = aidx + shift
                        if ((aidx_shifted >= 0) & (aidx_shifted <= max_num_atoms - 1)) :
                            split_4[j] = aidx_shifted
                        else:
                            split_4[j] = None
                    except:
                        # if a non-integer is in the wrong place of the enhanced stereo list, halt with None
                        return None
                split_4 = [str(s) for s in split_4 if s != None]
                if verbose == True :
                    print('Renumbered elements of group = ' + str(split_4))
                # if any valid index is left, the group is still valid, add it to new_enhanced_stereo_string
                if len(split_4) > 0 :
                    aidxs = ','.join(split_4)
                    new_enhanced_stereo_string = new_enhanced_stereo_string + grouptypeandnumber + ':' + aidxs + ','
                    if verbose == True :
                        print(new_enhanced_stereo_string)
                enh_st_index += 1
                
            # chop off the last comma from new_enhanced_stereo_string, if any, and enclose in '|'
            # also forming the final CXSMILES
            if len(new_enhanced_stereo_string) > 0 :
                cxsm_out = cxsm_out + ' |' + new_enhanced_stereo_string.rstrip(',') + '|'
            
    return cxsm_out

# Function that does the largest fragment picking by a different mechanism than the native rdkit one, taking a mol as input and issuing a mol as output

def largest_fragment_picker_mol(mol,
                                verbose = False,
                               ) :
    """
    Takes a mol as input.
    Identifies the molecular fragment with the largest number of atoms.
    Returns the mol of that fragment.
      
    Parameters
    ----------
    mol : rdkit molecule object
        The molecule to process.
    verbose : bool
        Defaults to False. Set to True *only for very small sets of compounds* to investigate the output more granularly.
        
    Returns
    -------
    mol
        The rdkit mol object of the largest fragment found in the input CXSMILES.
        Or None when anything went wrong during the process.
    """
    out = None
    try:
        mol_frags = rdmolops.GetMolFrags(mol, asMols = True)
        out = max(mol_frags, default = mol, key = lambda m : m.GetNumAtoms())
    except:
        if verbose == True :
            print('Some error was generated during the attempted largest fragment picking. Returning None.')
        pass
    
    return out

# Function that splits metal-nonmetal bonds heterolytically,
# only when both atoms are uncharged to start with, only for the cases foreseen by the ChEMBL pipeline.
# This is done to address the issue that ChEMBL will split such bonds generating multiple fragments,
# *after* we've already applied the keep largest fragment logic.

_alkoxide_pattern = Chem.MolFromSmarts("[Li,Na,K;+0]-[#7,#8;+0]")

def heterolytic_split_metal_nonmetal_bonds_like_ChEMBL(mol) :
    out = mol
    try:
        if mol.HasSubstructMatch(_alkoxide_pattern):            
            out = Chem.RWMol(mol)
            for match in out.GetSubstructMatches(_alkoxide_pattern):
                out.RemoveBond(match[0], match[1])
                out.GetAtomWithIdx(match[0]).SetFormalCharge(1)
                out.GetAtomWithIdx(match[1]).SetFormalCharge(-1)
    except:
        out = None
        pass
                
    return out

# Function that takes CXSMILES as input, does the standardisation and canonicalisation,
# converts the molecule back to CXSMILES.

def standardise_canonicalise_CXSMILES(CXSMILES,
                                      clear_stereo_from_output_CXSMILES = False,
                                      remove_mixtures = False,
                                      remove_non_organic_molecules = True,
                                      verbose = False,
                                     ) :
    """
    - Takes a (CX)SMILES as input.
    - Converts it to molecule.
      (--> any records that fail at this stage will return string 'ERROR: failed conversion of (CX)SMILES to molecule.')
    - Heterolytically splits any single bonds matching SMARTS pattern "[Li,Na,K;+0]-[#7,#8;+0]"
      (--> any molecules that cause an error in this step will return string 'ERROR: something went wrong when heterolytically splitting metal-nonmetal bonds.')
    - Removes mixtures if so requested.
      (--> any records that are mixtures will return string 'ERROR: input CXSMILES is a mixture and it was requested to remove mixtures.';
      if the mixture detection itself goes wrong, 'ERROR: failed CXSMILES mixture removal.')
    - Applies a customised 'largest_fragment_picker_mol' function to the molecule; necessary to circumvent a bug with rdkit's own
      rdMolStandardize.LargestFragmentChooser().choose() - see https://github.com/rdkit/rdkit/issues/6099
      (--> any records that fail at this stage will return string 'ERROR: failed CXSMILES largest fragment picking.')
    - Removes non-organic molecules, if so requested.
      (--> any record that is found not to be organic will return string 'ERROR: the largest fragment in the input CXSMILES is not organic and it was requested to remove non-organic molecules.';
      if the organic detection itself goes wrong, 'ERROR: the detection of organic nature of the molecule caused an unexpected error.')
    - Applies the customised rdkit and ChEMBL standardisation
      (--> any records that fail at this stage will return string 'ERROR: failed {...} standardisation.',
      where {...} stands for the specific process that failed.)
    - Applies canonicalisation (addressing 'or' stereo, meso chirality, single wedges set to RAC in each & group,
      presence of ABS groups only, standardisation of RAC groups @ strings).
      (--> any molecule with 'or' stereo is failed, and will return string 'ERROR: OR stereochemistry detected.')
    - Converts the molecule back to CXSMILES
      (--> any records that fail at this stage will return string 'ERROR: failed conversion of standardised molecule to (CX)SMILES.')    
      
    Parameters
    ----------
    CXSMILES : str
        The (CX)SMILES to process.
    clear_stereo_from_output_CXSMILES: bool
        Defaults to False. Set to True if you want to suppress all stereo in the output (tetrahedral, sp2...).
    remove_mixtures : bool
        defaults to False
        any input (CX)SMILES with '.' characters is a *mixture* of different fragments/molecules.
        the present version of this chemistry curation always keeps only the *largest fragment*.        
        - if your input (CX)SMILES are only fairly large organic molecules with small 'inert' salt or solvate fragments,
          it is usually safe to assume that activity/properties are mostly linked to the largest fragment,
          thus mixtures can be kept and will be stripped to the largest fragment.
        - if however you have mixtures where it is not sure which fragment is linked to activity/properties, or if all fragments are relevant,
          (e.g. unbuffered solubility of salts, or ionic inorganic species), then mixtures should be removed.
          --> in that case, set this parameter to True
          (yes, this means that at the moment this curation is NOT suitable for preparing data for ML on activity/properties of mixtures)
    remove_non_organic_molecules : bool
        defaults to True
        removes any record where the *largest fragment* does not contain any C-H, C-C or C-halogen bonds.
        this may be useful when wanting to focus on small molecule drugs (which is an implicit assumption of this curation).
        also NOTE: the largest fragment of inorganic molecules is often not the (only) relevant one for activity/properties.
    verbose : bool
        Defaults to False. Set to True *only for very small sets of compounds* to investigate errors more granularly.
    
    Returns
    -------
    str
        The standardised, canonicalised CXSMILES, or one of the error message strings (can be detected by its starting by 'ERROR').
    """    
    out = ''

    if verbose == True :
        print('Input CXSMILES = ' + CXSMILES)

    # make a molecule from the CXSMILES
    try:
        if verbose == True :
            print('> Converting CXSMILES input to molecule...')
        m = Chem.MolFromSmiles(CXSMILES)
    except:
        out = 'ERROR: failed conversion of (CX)SMILES to molecule.'
        return out
    if m == None:
        out = 'ERROR: failed conversion of (CX)SMILES to molecule.'
        return out

    # New: heterolytic split
    if verbose == True :
        print('> Heterolytically splitting alkoxide and similar metal-nonmetal bonds.')
    m = heterolytic_split_metal_nonmetal_bonds_like_ChEMBL(m)
    if m == None :
        out = 'ERROR: failed conversion of (CX)SMILES to molecule.'
        return out
   
    # Remove mixtures, if so requested
    if remove_mixtures == True :
        if verbose == True :
            print('> Detecting if the molecule is a mixture...')
        try:
            mol_frags = rdmolops.GetMolFrags(m, asMols = True)
            if len(mol_frags) > 1 :
                out = 'ERROR: input CXSMILES is a mixture and it was requested to remove mixtures.'
                if verbose == True :
                    print(out)
                return out
        except:
            out = 'ERROR: failed CXSMILES mixture removal.'
            if verbose == True :
                print(out)
            return out
            
    # New: apply a customised largest fragment picker (due to rdkit bug)
    try:
        if verbose == True :
            print('> Keeping largest fragment by rdmolops on molecule...')
        m = largest_fragment_picker_mol(m)
    except:
        out = 'ERROR: failed CXSMILES largest fragment picking.'
        return out
    if m == None :
        out = 'ERROR: failed CXSMILES largest fragment picking.'
        return out

    # Remove non-organic molecules, if so requested
    if remove_non_organic_molecules == True :
        if verbose == True :
            print('> Detecting if the largest fragment is organic...')
        organic = False
        try:
            m_org_det = Chem.AddHs(m)        
            for a in m_org_det.GetAtoms() :
                if a.GetSymbol() == 'C' :
                    for an in a.GetNeighbors() :
                        if an.GetSymbol() in ['C', 'H', 'F', 'Cl', 'Br', 'I'] :
                            organic = True
                            break
                if organic == True :
                    break
        except:
            out = 'ERROR: the detection of organic nature of the molecule caused an unexpected error.'
            if verbose == True :
                print(out)
            return out            
        if organic == False :
            out = 'ERROR: the largest fragment in the input CXSMILES is not organic and it was requested to remove non-organic molecules.'
            if verbose == True :
                print(out)
            return out
    
    if verbose == True :
        print('> Number of stereogroups in molecule = ' + str(len(m.GetStereoGroups())))

    # apply the rdkit and ChEMBL standardisation, plus canonicalisation    
    try:
        # DONE: after review, it was found that ChEMBL standardize_molblock removed stereo groups for molecules with > 1 fragment;
        # bug report created: https://github.com/chembl/ChEMBL_Structure_Pipeline/issues/49
        # --> addressed by ensuring that the molecule is already a single fragment at this point, see above (and removal of chooser below)
        
        # convert m to molblock (the ChEMBL pipeline standardizer requires that)
        # TODO: should be force a V3K molblock? --> probably not
        if verbose == True :
            print('> Converting molecule to MolBlock for ChEMBL pipeline..')
        mb = rdkit.Chem.rdmolfiles.MolToMolBlock(m)
        # standardize the molblock (ChEMBL pipeline)
        if verbose == True :
            print('> Standardising MolBlock using ChEMBL standardizer...')
        mb = standardizer.standardize_molblock(mb)
        # convert the standardized molblock back to molecule
        if verbose == True :
            print('> Converting standardised MolBlock back to molecule...')
        mol = Chem.rdmolfiles.MolFromMolBlock(mb)
        if verbose == True :
            print('> Number of stereogroups in ChEMBL-standardised molecule = ' + str(len(mol.GetStereoGroups())))
        
        # DONE: after review, it was found that this removes stereo groups for molecules with > 1 fragment;
        # bug report created: https://github.com/rdkit/rdkit/issues/6099
        # response by G. Landrum: duplicate of similar report from 2021; no solution offered
        # --> replaced with custom pre-standardisation largest fragment picker, see above
        # logic kept only for the record / in case rdkit fixes the bug at some point
        if False:
            # keep the largest fragment
            if verbose == True :
                print('> Keeping the largest fragment...')
            mlf = largest_Fragment.choose(mol)
            # if mlf is None, mark it as failed
            # otherwise mlf is the stripped, standardised molecule
            if verbose == True :
                print('> Number of stereogroups in largest fragment molecule = ' + str(len(mlf.GetStereoGroups())))
            if mlf == None:
                out = 'ERROR: failed molblock/largest fragment standardisation.'
                return out
        else:
            mlf = mol
        
        # convert to molblock
        if verbose == True :
            print('> Converting to MolBlock for ChEMBL checker...')        
        mb = rdkit.Chem.rdmolfiles.MolToMolBlock(mlf)
        # apply the ChEMBL checker
        if verbose == True :
            print('> Applying ChEMBL checker...')
        issue = checker.check_molblock(mb)
        #print(str(issue))
        if (len(str(issue)) >= 3):
            #if ((str(issue)[2] == "6") | (str(issue)[2] == "7")):
            # AFTER TESTING: decided to bypass error 6, or it kills all V3000 mols
            if str(issue)[2] == "7" :            
                out = 'ERROR: failed ChEMBL standardisation.'
                return out
            
        # Canonicalisation
        # 1.1. fail any molecule with 'or' stereochemistry
        # 1.2. set all meso stereocenters to abs
        # 1.3. flatten stereo when there is only 1 explicit sp3 stereo in each '&' group
        # 1.4. remove all stereogroups if there are *only* ABSOLUTE groups (--> CXSMILES == SMILES)
        # 1.5. invert the chirality of each AND group in order to have as many '@' as possible
        
        # This is only applicable if there is at least one stereogroup
        if len(mlf.GetStereoGroups()) == 0 :
            if verbose == True :
                print('> No stereogroups in molecule. Skipping stereogroup canonicalisation.')
        else :
            if verbose == True :
                print('> Applying stereogroup canonicalisation.')

            # 1.1.
            for group in mlf.GetStereoGroups() :
                if group.GetGroupType() == Chem.rdchem.StereoGroupType.STEREO_OR :
                    out = 'ERROR: OR stereochemistry detected.'
                    return out

            # Preparation for 1.2-5.
            # First make a dictionary of the current stereogroups, separated in ABSOLUTE and AND
            # by collecting their belonging atoms        
            stereogroup_atoms_dict = dict({'ABSOLUTE' : [], 'AND' : []})
            for group in mlf.GetStereoGroups() :
                if group.GetGroupType() == Chem.rdchem.StereoGroupType.STEREO_ABSOLUTE :                
                    stereogroup_atoms_dict['ABSOLUTE'].append([atom.GetIdx() for atom in group.GetAtoms()])
                if group.GetGroupType() == Chem.rdchem.StereoGroupType.STEREO_AND :                
                    stereogroup_atoms_dict['AND'].append([atom.GetIdx() for atom in group.GetAtoms()])

            if verbose == True :
                print('> Found ' + \
                      str(len(stereogroup_atoms_dict['ABSOLUTE'])) + \
                      ' ABSOLUTE groups and ' + \
                      str(len(stereogroup_atoms_dict['AND'])) + \
                      ' AND groups.')                    
                print('> Stereo groups atoms before this operation:')
                print(str(stereogroup_atoms_dict))
                
            # 1.2.
            # This is only applicable if there is at least one AND group
            if len(stereogroup_atoms_dict['AND']) > 0 :
                # For each STEREO_AND group, check if the atoms are meso stereocenters        
                # and if so, add them to the list of atoms to set to abs, and remove them from their original AND group                
                if verbose == True :
                    print('> Looking for meso stereoatoms in AND groups.')                
                current_SMILES = Chem.MolToSmiles(mlf)
                atoms_reset_to_abs = []
                # For each AND group, invert all its stereoatoms' chirality and check if the SMILES changes
                # (rdkit calculates the canonical SMILES, so meso fragments should always have the same sub-SMILES)
                for g_i, g_atomidx in enumerate(stereogroup_atoms_dict['AND']) :
                    # Make a RW version of the molecule
                    mlf_RW = Chem.RWMol(mlf)
                    # Invert all the stereocenters
                    for aidx in g_atomidx :
                        mlf_RW.GetAtomWithIdx(aidx).InvertChirality()
                    # Make the new SMILES
                    AND_inverted_SMILES = Chem.MolToSmiles(mlf_RW)
                    # Compare it with the old one, and if it is the same, reassign these atoms to an ABSOLUTE group
                    # create the new list of AND atoms in this group
                    if current_SMILES == AND_inverted_SMILES :                        
                        stereogroup_atoms_dict['AND'][g_i] = []
                        stereogroup_atoms_dict['ABSOLUTE'].append(g_atomidx)
                        atoms_reset_to_abs.extend(g_atomidx)
                if len(atoms_reset_to_abs) > 0 :
                    if verbose == True :
                        print('> ' + str(len(atoms_reset_to_abs)) + ' meso stereoatoms found, and reset to ABSOLUTE.')
                    # Delete any AND groups that have become empty
                    stereogroup_atoms_dict['AND'] = [l for l in stereogroup_atoms_dict['AND'] if len(l) > 0]                    
                    if verbose == True :
                        print('> Stereo groups atoms after meso correction: ')
                        print(str(stereogroup_atoms_dict))
                    
            # 1.3.
            # This is only applicable if there is at least one AND group (left)
            if len(stereogroup_atoms_dict['AND']) > 0 :
                if verbose == True :
                    print('> Looking for AND groups with only 1 stereoatom inside.')
                # Detect all STEREO_AND groups with only 1 atom inside;
                # flatten the atom ChiralType and remove it from the AND group
                for g_i, g_atomidx in enumerate(stereogroup_atoms_dict['AND']) :                    
                    g_atomidx = [int(x) for x in list(pd.unique(g_atomidx))]                    
                    if len(g_atomidx) == 1 :                        
                        if verbose == True :
                            print('> AND group ' + str(g_i) + ' has only 1 stereoatom. Flattening it and removing the AND group.')
                        a = mlf.GetAtomWithIdx(g_atomidx[0])
                        a.SetChiralTag(Chem.rdchem.ChiralType.CHI_UNSPECIFIED)
                        stereogroup_atoms_dict['AND'][g_i] = []
                # Delete any AND groups that have become empty
                stereogroup_atoms_dict['AND'] = [l for l in stereogroup_atoms_dict['AND'] if len(l) > 0]
                
            # 1.4.
            if len(stereogroup_atoms_dict['AND']) == 0 :
                if verbose == True :
                    print('> No AND groups left. Removing ABSOLUTE groups (which do not make sense on their own).')
                # There is no point keeping ABSOLUTE groups when there are no AND groups;
                # there is no enhanced stereochemistry to take into account
                stereogroup_atoms_dict['ABSOLUTE'] = []

            # 1.5.
            # This is only applicable if there is at least one AND group
            if len(stereogroup_atoms_dict['AND']) > 0 :
                # For each STEREO_AND group, invert all stereocenters and see if that results in a SMILES with more '@'s
                # and if so, replace the current molecule with the inverted one
                if verbose == True :
                    print('> Canonicalising @/@@ symbols for AND groups.')                
                groups_inverted = []
                # For each AND group, invert all its stereoatoms' chirality and check if the SMILES changes
                # (rdkit calculates the canonical SMILES, so meso fragments should always have the same sub-SMILES)
                # Calculate the current smiles and its number of isolated @@ and @ symbols
                current_SMILES = Chem.MolToSmiles(mlf)
                current_CCW = current_SMILES.count('@@')
                # This canonicalisation is only necessary if there are '@@' symbols in the SMILES
                if current_CCW != 0 :
                    current_CW = current_SMILES.count('@') - 2 * current_CCW
                    for g_i, g_atomidx in enumerate(stereogroup_atoms_dict['AND']) :
                        if verbose == True :
                            print('- current SMILES = ' + current_SMILES)
                            print('- AND group ' + str(g_i) +  ' : @@ = ' + str(current_CCW) + ', @ = ' + str(current_CW))
                        # Make a RW version of the molecule
                        mlf_RW = Chem.RWMol(mlf)
                        # Invert all the stereocenters in this AND group
                        for aidx in g_atomidx :
                            mlf_RW.GetAtomWithIdx(aidx).InvertChirality()
                        # Calculate the new smiles and its number of isolated @@ and @ symbols
                        inverted_SMILES = Chem.MolToSmiles(mlf_RW)
                        inverted_CCW = inverted_SMILES.count('@@')
                        # If all '@@' symbols have disappeared, replace the current molecule with the new one, and stop
                        if inverted_CCW == 0 :
                            groups_inverted.append(g_i)
                            mlf = mlf_RW
                            break
                        inverted_CW = inverted_SMILES.count('@') - 2 * inverted_CCW
                        # If the number of '@' symbols has increased, replace the current molecule with the new one
                        if inverted_CW > current_CW :
                            groups_inverted.append(g_i)
                            mlf = mlf_RW
                            current_SMILES = inverted_SMILES
                            current_CCW = inverted_CCW
                            current_CW = inverted_CW
                    if len(groups_inverted) > 0 :
                        if verbose == True :
                            print('> ' + str(len(groups_inverted)) + ' AND groups were inverted.')
                
            if verbose == True :
                print('> Applying the resulting stereogroups to the molecule.')

            # Remake the groups after the above changes
            stereo_groups_list = []                
            # - first make the molecule writable
            mlf_RW = Chem.rdchem.RWMol(mlf)
            # Remove all current stereogroups, to avoid issues
            mlf_RW.SetStereoGroups(stereo_groups_list)
            # Create the ABSOLUTE group, if any
            atoms_for_ABS_group = []
            for g_atomidx in stereogroup_atoms_dict['ABSOLUTE'] :
                atoms_for_ABS_group.extend(g_atomidx)
            atoms_for_ABS_group = [int(x) for x in list(pd.unique(atoms_for_ABS_group))]
            if len(atoms_for_ABS_group) > 0 :                
                abs_group = Chem.CreateStereoGroup(
                    Chem.rdchem.StereoGroupType.STEREO_ABSOLUTE,
                    mlf_RW,
                    atoms_for_ABS_group)
                stereo_groups_list.append(abs_group)
            # Create the AND groups, if any
            for g_atomidx in stereogroup_atoms_dict['AND'] :
                # uniqueness already guaranteed from 1.3.
                and_group = Chem.CreateStereoGroup(
                    Chem.rdchem.StereoGroupType.STEREO_AND,
                    mlf_RW,
                    g_atomidx)
                stereo_groups_list.append(and_group)
            # - assign the new stereo groups to the RW molecule
            mlf_RW.SetStereoGroups(stereo_groups_list)
            # Copy to the RO molecule
            mlf = mlf_RW
            
    except:        
        # if anything has gone wrong above, also mark the molecule as failed
        out = 'ERROR: failed standardisation.'
        return out

    # convert the standardised molecule back to CXSMILES
    try:           
        if verbose == True :
            print('> Calculating the CXSMILES.')
        
        smilesWriterParams = Chem.SmilesWriteParams()
        smilesWriterParams.canonical = True
        smilesWriterParams.isomericSmiles = not(clear_stereo_from_output_CXSMILES)
        cxsm = Chem.MolToCXSmiles(mlf,
                                  smilesWriterParams,
                                  Chem.rdmolfiles.CXSmilesFields.CX_ENHANCEDSTEREO)        
        if cxsm == None:
            out = 'ERROR: failed conversion of standardised molecule to (CX)SMILES.'
            return out
    except:
        out = 'ERROR: failed conversion of standardised molecule to (CX)SMILES.'
        return out

    return cxsm

# Function that curates chemistry and/or data of an unpivoted csv
def F3_csv_unpivoted_to_standard_transformed_curated(
    input_unpivoted_csv_file_path,
    output_unpivoted_csv_file_path,
    original_ID_colname = 'ID',
    do_chemistry_curation = True,
    SMILES_colname = 'SMILES',    
    remove_mixtures = False,
    remove_non_organic_molecules = True,
    clear_stereo_from_output_SMILES = False,
    do_data_curation = True,
    MODULE_colname = 'MODULE',
    MODULE_rename_dict = None,
    VALUE_colname = 'VALUE',
    VALUE_prefix_colname = None,
    TRANSF_dict = {'fumic' : 'log10(x/(1-x))', 'ppb' : 'log10(x/(100-x))', 'log' : 'x'},    
    TRANSF_default = 'log10(x)',
    transform_censored = True,
    ):
    """
    - Takes a file as input, with structures encoded as (CX)SMILES, and/or data.
    - Does chemistry and/or data curation, as requested.
    - Saves a standardised curated file with standard column names:
      ['original_ID', 'SMILES', 'MODULE', 'VALUE', 'TRANSFORMATION', 'TRANSFORMED_VALUE']
      (minus the ones that don't apply if chemistry and/or data curation is skipped).
    
    Parameters
    ----------
    input_unpivoted_csv_file_path : str
        path to the input csv file, can be (g)zipped
    output_unpivoted_csv_file_path : str
        path to the output csv file, can end in .csv, .csv.gz or .csv.zip    
    original_ID_colname : str
        name of the compound ID column, defaults to 'ID', mandatory
        if column original_ID_colname is in the input file, it must be defined in all rows
        if column original_ID_colname is NOT in the input file, then:
            if do_chemistry_curation == True, the *original* (NOT the curated) SMILES will be copied to it
            if do_chemistry_curation == False, the record index will be copied to it
    do_chemistry_curation : bool
        defaults to True, set to False if only data curation is needed
    SMILES_colname : str
        name of the (CX)SMILES-containing column, mandatory
        must be defined in all rows in the file
        records for which the SMILES is invalid:
        -->  saved in output_unpivoted_csv_file_path + '_invalid_SMILES.csv.gz'
        records for which the SMILES failed the rdkit + ChEMBL standardisation
        -->  saved in output_unpivoted_csv_file_path + '_failed_std_SMILES.csv.gz'
    remove_mixtures : bool
        defaults to False
        any input (CX)SMILES with '.' characters is a *mixture* of different fragments/molecules.
        the present version of this chemistry curation always keeps only the *largest fragment*.        
        - if your input (CX)SMILES are only fairly large organic molecules with small 'inert' salt or solvate fragments,
          it is usually safe to assume that activity/properties are mostly linked to the largest fragment,
          thus mixtures can be kept and will be stripped to the largest fragment.
        - if however you have mixtures where it is not sure which fragment is linked to activity/properties, or if all fragments are relevant,
          (e.g. unbuffered solubility of salts, or ionic inorganic species), then mixtures should be removed.
          --> in that case, set this parameter to True
          (yes, this means that at the moment this curation is NOT suitable for preparing data for ML on activity/properties of mixtures)
    remove_non_organic_molecules : bool
        defaults to True
        removes any record where the *largest fragment* does not contain any C-H, C-C or C-halogen bonds.
        this may be useful when wanting to focus on small molecule drugs (which is an implicit assumption of this curation).
        also NOTE: the largest fragment of inorganic molecules is often not the (only) relevant one for activity/properties.
    clear_stereo_from_output_SMILES : bool
        defaults to False
        should the SMILES be cleared of sp3 and sp2 stereo features?
        this is sometimes used in modelling (in which case it must be done at this stage, before averaging)    
    do_data_curation : bool
        defaults to True, set to False if only chemistry curation is needed
    MODULE_colname : str
        defaults to 'MODULE'
        mandatory string; must be defined in all rows in the file
        NEW: MODULE_colname can be a *list* of column names; all must be defined in all rows in the file.
        > MODULE in the output will then be a concatenation of the contents of these columns, in the same order.
          Separator = '_' in simple string pasting mode (so 'ACT_X' and 'IC50' will become 'ACT_X_IC50' --> irreversible).
    MODULE_rename_dict : dict
        defaults to None
        optional dictionary, if you want to rename the original values in the MODULE_colname column to standard ones
        NOTE: matches must be exact (case-sensitive)! Any original name that is not found will not be changed.
    VALUE_colname : str
        defaults to 'VALUE'
        mandatory string, must be defined in all rows in the file
    VALUE_prefix_colname : str or None
        defaults to None
        optional string; if given, must be defined in all rows in the file
        > to use if VALUE_colname contains numerical values, and VALUE_prefix_colname contains qualifiers,
          e.g. 5 and '<', respectively, which will result in an output VALUE equal to '<5'.
          NOTE: only prefix '=' is omitted. Do NOT assume NaN for VALUE_colname values that are not qualified.
    TRANSF_dict : dict
        defaults to {'fumic' : 'log10(x/(1-x))', 'ppb' : 'log10(x/(100-x))', 'log' : 'x'},
        heuristic dictionary of transformations to apply to the VALUE's, based on MODULE's (substring) matches
        > first, an exact (and case-sensitive) match of each MODULE to the keys in TRANSF_dict is attempted
        > if no exact match is found, then a case-insensitive substring match is attempted:
          {substring_1 : transformation_string_1, substring_2 : transformation_string_2, ...}        
        > if a MODULE.lower() contains a given substring, the corresponding transformation is applied
        > allowed transformation_string values : 'x', 'log10(x)', 'log10(x/(1-x))', 'log10(x/(100-x))'
          NOTE: 'x' means no transformation at all, i.e. VALUE is not processed (e.g. not even string '5.3' to float 5.3)
        > subtrings order: more specific to less specific (as the first found is applied)
          (e.g. put 'fumic' before 'ppb' if the Fumic MODULE contains both 'fumic' and 'ppb')
        > leave empty {} if you don't want to transform anything (e.g. for text), and set TRANSF_DEFAULT to 'x'
        NOTE: MODULE_rename_dict, if provided, is applied *before* TRANSF_dict! So use the *renamed* MODULE names in TRANSF_dict.
    TRANSF_default : str
        defaults 'log10(x)'
        fallback default transformation (for anything not covered by the TRANSF_dict)
    transform_censored : bool
        defaults to True
        should values like '<0.01' and '>1000' be transformed?
        (e.g. if the transformation is 'log10(x)', to '<-2' and '>3', respectively)
        NOTE: if set to False, censored values will be set to NA !
    
    Returns
    -------
    None
        Saving the resulting standardised unpivoted file is the only expected action by this function.
    """

    # First check that the curation requirements make sense
    if ((do_chemistry_curation == False) & (do_data_curation == False)):
        raise ValueError('It is not being requested to do either chemistry curation or data curation. This function has nothing to do. Please review your parameters.')        
    
    # Define the relevant columns    
    relevant_colnames = [original_ID_colname]    
    if do_chemistry_curation == True :
        relevant_colnames.append(SMILES_colname)    
    if do_data_curation == True :
        relevant_colnames.append(VALUE_colname)    
        if isinstance(MODULE_colname, list):
            relevant_colnames.extend(MODULE_colname)
        else:
            relevant_colnames.append(MODULE_colname)
        if VALUE_prefix_colname != None:
            relevant_colnames.append(VALUE_prefix_colname)
        # Sanity check: transformations in the heuristic dictionary and default must be valid
        for t in list(TRANSF_dict.values()) + [TRANSF_default] :
            if t not in ['x', 'log10(x)', 'log10(x/(1-x))', 'log10(x/(100-x))'] :
                raise ValueError("Invalid TRANSF. Choose from 'x', 'log10(x)', 'log10(x/(1-x))', 'log10(x/(100-x))'")
                
    # Read the unpivoted data

    print("")
    print("Reading file " + input_unpivoted_csv_file_path)
    print("")
    
    # Collect the column names of the input file
    firstrow = pd.read_csv(input_unpivoted_csv_file_path, nrows = 0)
    cols_to_read = [c for c in firstrow.columns if c in relevant_colnames]
    
    if len(cols_to_read) == 0 :
        raise ValueError('None of the mandatory columns are present in the input file. Please review your data.')

    if (do_chemistry_curation == True) & (SMILES_colname not in cols_to_read) :
        raise ValueError(SMILES_colname + ' is not present in the input file, despite the request to curate chemistry. Please review your data.')
    
    if do_data_curation == True :
        if VALUE_colname not in cols_to_read :
            raise ValueError(VALUE_colname + ' is not present in the input file, despite the request to curate data. Please review your data.')        
        if isinstance(MODULE_colname, list):
            if any([c not in cols_to_read for c in MODULE_colname]):
                raise ValueError('At least one of ' + str(MODULE_colname) + ' is not present in the input file, despite the request to curate data. Please review your data.')
        else:
            if MODULE_colname not in cols_to_read:
                raise ValueError(MODULE_colname + ' is not present in the input file, despite the request to curate data. Please review your data.')
    
    # Read the full file
    df_unpiv = pd.read_csv(input_unpivoted_csv_file_path,
                           usecols = cols_to_read,
                           na_values = '', keep_default_na = False)

    # Reset the index, in case it was imposed
    df_unpiv.reset_index(drop = True, inplace = True)
    
    # If the ID column is not defined, handle it depending on which curation is requested
    if original_ID_colname not in list(df_unpiv.columns) :
        if do_chemistry_curation == True :
            df_unpiv[original_ID_colname] = df_unpiv[SMILES_colname].copy()
        else :
            df_unpiv[original_ID_colname] = list(df_unpiv.index)
        
    # Check that the mandatory columns are all present and dense
    for c in relevant_colnames :
        if c not in df_unpiv.columns :
            raise ValueError('Column ' + c + ' is not present in the input file. Please review your data.')
        if any(df_unpiv[c].isna()) :
            raise ValueError('Column ' + c + ' is NA in at least one row. Please review your data.')

    if do_chemistry_curation == True :
        # Check that no different SMILES exist for single ID's
        N_unique_SMILES_per_ID = df_unpiv.groupby(original_ID_colname)[SMILES_colname].nunique()
        IDs_non_unique = [str(ID) for ID in N_unique_SMILES_per_ID[N_unique_SMILES_per_ID != 1].index]
        if len(IDs_non_unique) > 0 :
            raise ValueError(original_ID_colname + ' = [' + ','.join(IDs_non_unique) + '] \
                each corresponds to more than one distinct ' + SMILES_colname + '. \
                This is a data integrity breach. Please review your data.')

    if do_data_curation == True :
        # If MODULE_colname is a list of columns, create the '_'.joined MODULE column from it, using the first colname as output
        if isinstance(MODULE_colname, list):
            df_unpiv[MODULE_colname[0]] = df_unpiv.apply(lambda row : '_'.join([str(c) for c in row[MODULE_colname]]), axis = 1)
            # Reassign the MODULE column name to only the first one that we just replaced with the concatenation, so it is the only one kept
            MODULE_colname = MODULE_colname[0]

        # Create qualified values using the prefix column, if applicable, and replace the VALUE_colname with the result
        if VALUE_prefix_colname != None:
            df_unpiv[VALUE_colname] = [v if q == '=' else str(q) + str(v) for v, q in zip(df_unpiv[VALUE_colname], df_unpiv[VALUE_prefix_colname])]

    # Reorder the columns and rename them to the 'standard' names
    if ((do_chemistry_curation == True) & (do_data_curation == True)):
        reordered_list_cols_to_output = [SMILES_colname, original_ID_colname, MODULE_colname, VALUE_colname]
    elif ((do_chemistry_curation == False) & (do_data_curation == True)):
        reordered_list_cols_to_output = [original_ID_colname, MODULE_colname, VALUE_colname]
    elif ((do_chemistry_curation == True) & (do_data_curation == False)):
        reordered_list_cols_to_output = [SMILES_colname, original_ID_colname]
    
    df_unpiv = df_unpiv[reordered_list_cols_to_output]
        
    df_unpiv.rename(columns = {SMILES_colname : 'SMILES',
                               original_ID_colname : 'original_ID',
                               MODULE_colname : 'MODULE',
                               VALUE_colname : 'VALUE'},
                               inplace = True,
                               errors = 'ignore')
    
    print("Found " + str(df_unpiv.shape[0]) + " records.")
    print("")

    # A. Curate the chemistry

    if do_chemistry_curation == True :
    
        print("Chemistry Curation")
        print("------------------")
        print("")

        # create a dictionary of unique SMILES vs the indices of df_unpiv they map to
        index_vs_SMILES_dict = dict()
        for i, s in zip(df_unpiv.index, df_unpiv['SMILES']) :
            if s not in index_vs_SMILES_dict :
                index_vs_SMILES_dict[s] = [i]
            else:
                index_vs_SMILES_dict[s].append(i)

        print("Found " + str(len(index_vs_SMILES_dict.keys())) + " unique SMILES.")
        print("")
        
        print("> converting SMILES to molecules and applying rdkit + ChEMBL + standardisation...")
        print("")

        # DONE for v14: encoded the 'pure' standardisation part of this process into a separate function
        #               ('standardise_canonicalise_CXSMILES')
        #               taking a CXSMILES as input and outputting the standardised CXSMILES if OK, or
        #               error text, starting by 'ERROR'
        invalid_SMILES_index = []
        failed_std_SMILES_index = []
        std_SMILES_vs_index = [''] * df_unpiv.shape[0]    
        for s, ix in zip(index_vs_SMILES_dict.keys(), index_vs_SMILES_dict.values()) :
            try:
                # Apply the CXSMILES standardisation function
                std_canon_CXSMILES = standardise_canonicalise_CXSMILES(s,
                                                                       clear_stereo_from_output_CXSMILES = clear_stereo_from_output_SMILES,
                                                                       remove_mixtures = remove_mixtures,
                                                                       remove_non_organic_molecules = remove_non_organic_molecules,
                                                                       verbose = False
                                                                      )
                # Case 1: invalid input CXSMILES
                if (std_canon_CXSMILES == 'ERROR: failed CXSMILES largest fragment picking.') | \
                   (std_canon_CXSMILES == 'ERROR: failed conversion of (CX)SMILES to molecule.') :
                    invalid_SMILES_index.extend(ix)
                # Case 2: other fail (rdkit or ChEMBL or specific rules)
                elif std_canon_CXSMILES.startswith('ERROR') :
                    failed_std_SMILES_index.extend(ix)
                # Case 3: all fine, valid CXSMILES obtained
                else :
                    # Assign the standardised CXSMILES to the appropriate index positions
                    for i in ix :
                        std_SMILES_vs_index[i] = std_canon_CXSMILES
            except:
                # if anything unexpected has gone wrong above, also mark the molecule as failed
                failed_std_SMILES_index.extend(ix)
    
        del index_vs_SMILES_dict

        # Report and drop any invalid CXSMILES records
        if len(invalid_SMILES_index) > 0 :
            print("")
            print("Found " + str(len(invalid_SMILES_index)) + ' invalid (CX)SMILES.')
            print("")
            invalid_SMILES_file_path = output_unpivoted_csv_file_path + '_invalid_SMILES.csv.gz'
            print("Saving corresponding records to: " + invalid_SMILES_file_path)
            df_unpiv, df_unpiv_inv_SM = df_unpiv.drop(invalid_SMILES_index, axis = 0), df_unpiv.loc[invalid_SMILES_index]
            df_unpiv_inv_SM.to_csv(invalid_SMILES_file_path, index = False)

        # Report and drop any failed standardisation CXSMILES records
        if len(failed_std_SMILES_index) > 0 :
            print("")
            print("Found " + str(len(failed_std_SMILES_index)) + ' (CX)SMILES that did not pass the rdkit + standardisation and curation.')
            print("")
            failed_std_SMILES_file_path = output_unpivoted_csv_file_path + '_failed_std_SMILES.csv.gz'
            print("Saving to: " + failed_std_SMILES_file_path)
            df_unpiv, df_unpiv_fail_SM = df_unpiv.drop(failed_std_SMILES_index, axis = 0), df_unpiv.loc[failed_std_SMILES_index]
            df_unpiv_fail_SM.to_csv(failed_std_SMILES_file_path, index = False)

        if df_unpiv.shape[0] == 0 :
            raise ValueError('After validating and standardising the SMILES, no records are left.\
                There are no valid records, the process cannot continue.')

        # Map the standardised CXSMILES back to df_unpiv
        df_unpiv['SMILES'] = [std_SMILES_vs_index[i] for i in df_unpiv.index]

    # B. Curate the data

    if do_data_curation == True :
        
        print("")
        print("Data Curation")
        print("-------------")
        print("")

        # create a list of unique MODULE's
        unique_MODULES = list(df_unpiv['MODULE'].unique())

        # if a MODULE renaming dictionary is provided, apply it
        if MODULE_rename_dict != None :
            # first validate the existing MODULE names
            MODULE_renaming = dict()
            for m in unique_MODULES :
                if m in MODULE_rename_dict :
                    MODULE_renaming[m] = MODULE_rename_dict[m]
                else :
                    MODULE_renaming[m] = m
            # then apply the renaming
            unique_MODULES = [MODULE_renaming[m] for m in unique_MODULES]
            df_unpiv['MODULE'] = [MODULE_renaming[m] for m in df_unpiv['MODULE']]
        
        # Create a transformation dictionary for the unique MODULE's in df_unpiv,
        # based on TRANSF_dict and TRANSF_default
        TRANSF_dict_MODULE = dict()
        for p in unique_MODULES:
            TRANSF_dict_MODULE[p] = TRANSF_default
            # First try an exact match
            try:
                TRANSF_dict_MODULE[p] = TRANSF_dict[MODULE[p]]
            except:
                # Otherwise try a case-insensitive substring match, stopping at the first found
                for k in TRANSF_dict.keys():
                    if str(k).lower() in str(p).lower():
                        TRANSF_dict_MODULE[p] = TRANSF_dict[k]
                        break

        # Map the TRANSFORMATION's onto df_unpiv
        df_unpiv['TRANSFORMATION'] = [TRANSF_dict_MODULE[m] for m in df_unpiv['MODULE']]

        # define utility function that determines if an object (string or float)
        # is a float or censored float AND is not infinite or NA;
        # if so, returns a list with the uncensored value and the qualifier
        #  ('<' or '>' for censored, pd.NA for floats)
        # in all other cases, returns [pd.NA, pd.NA]
        def uncensor_float(v):
            try:
                v = v.lstrip()
            except:
                pass
            u = pd.NA
            c = '='
            try:
                u = float(v)
                if math.isinf(u) :
                    u = pd.NA
            except:
                try:
                    v = str(v)
                    uf = float(v[1:])
                    cf = v[0]
                    if ((cf == '<') | (cf == '>')) & (not(math.isinf(uf) | pd.isna(uf))):
                        u = uf
                        c = cf
                except:
                    pass
            return [u, c]

        # Make [value, qualifier] lists for all valid records
        # and store them into temporary columns 'V_u' and 'V_c'
        uc = df_unpiv['VALUE'].map(uncensor_float)
        u, c = [], []
        for uci in uc:
            u.append(uci[0])
            c.append(uci[1])
        df_unpiv['V_u'] = u
        df_unpiv['V_c'] = c

        # define utility function that applies the indicated transformation
        # or returns an NA, for an individual *numerical* value x.
        # (No need to handle TRANSF = 'x'; that is bypassed below).
        def transform_value(x, TRANSF):
            out = pd.NA
            if pd.notna(x):
                    if ((TRANSF == 'log10(x)') & (x > 0)) | \
                        ((TRANSF == 'log10(x/(1-x))') & ((x > 0) & (x < 1))) | \
                        ((TRANSF == 'log10(x/(100-x))') & ((x > 0) & (x < 100))) :
                        out = eval(TRANSF)            
            return out

        print("> transforming VALUE's, by applying the provided functions...")
        print("")

        # Store the transformed values in temporary column 'TV_u'
        df_unpiv['TV_u'] = [transform_value(u, t) if t != 'x' else v for u, v, t in \
            zip(df_unpiv['V_u'], df_unpiv['VALUE'], df_unpiv['TRANSFORMATION'])]

        # Make the TRANSFORMED_VALUE column, by pasting the qualifier V_c before the transformed value TV_u,
        # unless the qualifier is '=', or the transformed value itself is NA, or the transformation is 'x';
        # in all other cases, just copy the transformed value TV_u as such to TRANSFORMED_VALUE
        df_unpiv['TRANSFORMED_VALUE'] = [c + str(tv) if ((t != 'x') & (c != '=') & (pd.notna(tv))) \
                else tv for tv, c, t in \
            zip(df_unpiv['TV_u'], df_unpiv['V_c'], df_unpiv['TRANSFORMATION'])]

        # If transform_censored is False, set to pd.NA all TRANSFORMED_VALUE's
        # for which V_c is not '=' and the transformation is not 'x'
        if transform_censored == False :
            df_unpiv.loc[(df_unpiv['V_c'] != '=') & (df_unpiv['TRANSFORMATION'] != 'x'), 'TRANSFORMED_VALUE'] = pd.NA

        # Drop the temporary columns
        df_unpiv.drop(columns = ['V_u', 'V_c', 'TV_u'], inplace = True, errors = 'ignore')
    
    print("")
    print("All curation processes completed.")
    print("")
    print("Saving final records to : " + output_unpivoted_csv_file_path)
    df_unpiv.to_csv(output_unpivoted_csv_file_path, index = False)

def F4_csv_unpivoted_std_transf_cur_to_averaged(
    input_curated_unpivoted_csv_file_path,
    output_averaged_unpivoted_csv_file_path,
    keep_only_numeric = False,
    keep_uncensored_and_qualifier_in_output = False,
    min_number_data_points = 0,
    aggregation_function = 'mean',
    remove_outliers = False,
    ):
    """
    - Takes an *unpivoted* *standardised* file as input, as resulting from running F3_csv_unpivoted_to_standard_transformed_curated.
    - Standard columns are expected: ['original_ID', 'SMILES', 'MODULE', 'VALUE', 'TRANSFORMATION', 'TRANSFORMED_VALUE']; any other column is ignored.
    - Aggregates records by the 'SMILES' column, and applies the specified averaging function to the 'TRANSFORMED_VALUE' column values.
    - Sorted unique values from the 'original_ID' column are kept, for each aggregated set of records, and output in 'original_ID' in the output, ';'-separated.
    - Saves a standardised curated averaged file with standard column names:
      ['original_ID', 'SMILES', 'MODULE', 'VALUE_AVG', 'TRANSFORMATION', 'TRANSFORMED_VALUE_N']
      +
       ['TRANSFORMED_VALUE_MEAN', 'TRANSFORMED_VALUE_SAMPLE_SD'] if aggregation_function == 'mean'
      or
       ['TRANSFORMED_VALUE_MEDIAN', 'TRANSFORMED_VALUE_SAMPLE_IQR'] if aggregation_function = 'median'
      +
       ['TRANSFORMED_VALUE_{x}_U', 'TRANSFORMED_VALUE_{x}_C'] only if  keep_uncensored_and_qualifier_in_output == True; x = aggregation_function.upper()
    
    Parameters
    ----------
    input_curated_unpivoted_csv_file_path : str
        - path to the input standardised, curated csv file, can be (g)zipped
    output_averaged_unpivoted_csv_file_path : str
        - path to the output csv file, can end in .csv, .csv.gz or .csv.zip    
    keep_only_numeric : bool
        - defaults to False
        - set to True if you want to remove censored *transformed* values
          (like '<-2' or '>3'); NA's are always removed by default
          > any records that are discarded due to this choice will be saved in: input_curated_unpivoted_csv_file_path + '_NA_or_excluded_from_averaging.csv.gz'    
    keep_uncensored_and_qualifier_in_output : bool
        - defaults to False
        - set to True if you want to keep TRANSFORMED_VALUE_MEAN_U (e.g. 5) and TRANSFORMED_VALUE_MEAN_C (e.g. '>') when TRANSFORMED_VALUE_MEAN = '>5' in the output.
    min_number_data_points : integer
        - defaults to 0
        - set to an integer > 0 if you want to filter out any MODULE for which fewer than these data points exist, after averaging
          > any records that are discarded due to this choice will be saved in: input_curated_unpivoted_csv_file_path + '_averged_too_few_data_points.csv.gz'
    aggregation_function : str
        - defaults to 'mean'
        - or set to 'median'
          > if 'mean': TRANSFORMED_VALUE_MEAN and TRANSFORMED_VALUE_SAMPLE_SD will be output
          > if 'median', TRANSFORMED_VALUE_MEDIAN and TRANSFORMED_VALUE_IQR will be output.
    remove_outliers : bool
        - defaults to False
        - set to True if you want the function to identify outliers (i.e. values not included in [Q1 - 1.5 * IQR, Q3 + 1.5 * IQR]) and remove them from the calculation of the mean
        - only applies when aggregation_function == 'mean'; the 'median' method already picks a robust value by definition.
    
    Returns
    -------
    None
        Saving the resulting standardised unpivoted file is the only expected action by this function.
    """
    
    # v14: we suppress the option to merge by ID and SMILES; only CXSMILES is used for merging from now on    
    # rather that rewriting all the logic, we simply set to False the parameter that controlled this behaviour
    keep_distinct_original_ID_per_SMILES = False
    
    # Read the data
    print("")
    print("Reading file " + input_curated_unpivoted_csv_file_path)    

    df_cur = pd.read_csv(input_curated_unpivoted_csv_file_path, usecols = ['original_ID', 'SMILES', 'MODULE', 'TRANSFORMED_VALUE', 'TRANSFORMATION'])

    # Reset index, in case it was imposed
    df_cur.reset_index(drop = True, inplace = True)

    # First, sanity check: there should be only one TRANSFORMATION per MODULE
    df_cur_M_T = df_cur[['MODULE', 'TRANSFORMATION']].drop_duplicates()

    if any(df_cur_M_T.groupby('MODULE')['TRANSFORMATION'].count() > 1):
        raise ValueError("There appears to be at least one MODULE for which different TRANSFORMATION's are defined.\nThis is not allowed. Please review your data/process.")

    # If the sanity check is passed, created a dictionary of reverse transformations per MODULE
    all_transfs = ["x", "log10(x)", "log10(x/(1-x))", "log10(x/(100-x))"]
    # Although the file processed by F4 should come from F3, thus only have allowed transformations, check anyway
    if any([m not in all_transfs for m in df_cur_M_T['TRANSFORMATION']]):
        raise ValueError("There appears to be at least one TRANSFORMATION that is not in the allowed list:\n" + ','.join(all_transfs) + "\nPlease review your data/process.")
    all_reverse_transf_functs = ["x", "exp(x*log(10))", "1/(1+exp(-1*x*log(10)))", "100/(1+exp(-1*x*log(10)))"]
    transf_to_reverse_transf_functs = dict(zip(all_transfs, all_reverse_transf_functs))

    # Identify numerical and censored values

    # define utility function that determines if an object (string or float)
    # is a float or censored float AND is not infinite or NA;
    # if so, returns a list with the uncensored value and the qualifier
    #  ('<' or '>' for censored, pd.NA for floats)
    # in all other cases, returns [pd.NA, pd.NA]
    def uncensor_float(v):
        try:
            v = v.lstrip()
        except:
            pass
        u = pd.NA
        c = '='
        try:
            u = float(v)
            if math.isinf(u) :
                u = pd.NA
        except:
            try:
                v = str(v)
                uf = float(v[1:])
                cf = v[0]
                if ((cf == '<') | (cf == '>')) & (not(math.isinf(uf) | pd.isna(uf))):
                    u = uf
                    c = cf
            except:
                pass
        return [u, c]

    print("")
    print("Total # records = " + str(df_cur.shape[0]))
    print("")
    
    # Make [value, qualifier] lists for all valid records
    # and store them into temporary columns 'V_u' and 'V_c'
    uc = df_cur['TRANSFORMED_VALUE'].map(uncensor_float)
    u, c = [], []
    for uci in uc:
        u.append(uci[0])
        c.append(uci[1])
    df_cur['V_u'] = u
    df_cur['V_c'] = c

    notna_values_bool = df_cur['V_u'].notna()

    if keep_only_numeric == True:
        print("> filtering to only non-NA, numerical transformed values...")        
        notqualifier_values_bool = (df_cur['V_c'] == '=')
        valid_recs_bool = [(nv & nq) for nv, nq in zip(notna_values_bool, notqualifier_values_bool)]
        valid_recs_index = notna_values_bool[valid_recs_bool].index
        df_invalid, df_cur = df_cur.drop(valid_recs_index, axis = 0), df_cur.loc[valid_recs_index]
    else:
        print("> filtering to only non-NA, numerical and '<'/'>'-censored transformed values...")
        valid_recs_index = notna_values_bool[notna_values_bool == True].index
        df_invalid, df_cur = df_cur.drop(valid_recs_index, axis = 0), df_cur.loc[valid_recs_index]        

    if df_invalid.shape[0] > 0 :
        print("")
        print("Saving records that are NA or excluded from averaging to : " + input_curated_unpivoted_csv_file_path + '_NA_or_excluded_from_averaging.csv.gz')
        df_invalid.to_csv(input_curated_unpivoted_csv_file_path + '_NA_or_excluded_from_averaging.csv.gz', index = False)

    if df_cur.shape[0] == 0:
        raise ValueError("After dropping records that are NA or excluded from averaging, nothing is left. Please review your parameters/data.")

    print("")
    print("Found " + str(df_cur.shape[0]) + " valid records.")

    # Average by groups
    # Define a function that finds the most frequent qualifier, if any
    def most_freq_qualif(pdSeries):
        c_list = list(pdSeries)
        c = '='
        if '<' in c_list :
            c = '<'
        if '>' in c_list :
            Nlt = c_list.count('<')
            Ngt = c_list.count('>')            
            c = sorted(zip([Nlt, Ngt], ['<','>']))[1][1]
        return c

    if aggregation_function == 'mean':
        dispersion_agg = 'std'
        TRANSFORMED_VALUE_CENTRAL_TENDENCY_colname = 'TRANSFORMED_VALUE_MEAN'
        TRANSFORMED_VALUE_DISPERSION_colname = 'TRANSFORMED_VALUE_SAMPLE_SD' 
    elif aggregation_function == 'median':
        def dispersion_agg(x):
            q1 = x.quantile(0.25)
            q3 = x.quantile(0.75)
            return q3 - q1
        TRANSFORMED_VALUE_CENTRAL_TENDENCY_colname = 'TRANSFORMED_VALUE_MEDIAN'
        TRANSFORMED_VALUE_DISPERSION_colname = 'TRANSFORMED_VALUE_IQR' 
    else:
        raise ValueError('The only allowed aggregation_function values are "mean" and "median".')

    # New in v18.2 : remove outliers, if so requested
    if remove_outliers == True :
        # Define a function that takes returns the lower and upper outlier bounds
        def find_outliers(group):
            x = group['V_u']
            Q1 = x.quantile(0.25)
            Q3 = x.quantile(0.75)            
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR    
            return pd.Series({'lower_bound' : lower_bound, 'upper_bound' : upper_bound})

        # Create a df with the lower and upper bounds for each group
        df_cur_obs = df_cur.groupby(['original_ID', 'SMILES', 'MODULE', 'TRANSFORMATION']).apply(find_outliers, include_groups = False).reset_index()
        # Merge the info into df_cur
        df_cur = df_cur.merge(df_cur_obs, how = 'left')
        # Use the info to filter to only non-outlier cases
        df_cur = df_cur.loc[((df_cur['V_u'] >= df_cur['lower_bound']) & (df_cur['V_u'] <= df_cur['upper_bound']))]
        # Remove the lower and upper bound columns
        del df_cur['lower_bound']
        del df_cur['upper_bound']
            
    print("")
    if keep_distinct_original_ID_per_SMILES == True :
        print("Averaging valid TRANSFORMED_VALUE's after grouping by original_ID, SMILES, MODULE, TRANSFORMATION.")
        if keep_only_numeric == False:
            df_cur = df_cur.groupby(['original_ID', 'SMILES', 'MODULE', 'TRANSFORMATION'],
                        as_index = False, sort = False).agg(                    
                        TRANSFORMED_VALUE_CENTRAL_TENDENCY = pd.NamedAgg(column = 'V_u', aggfunc = aggregation_function),
                        TRANSFORMED_VALUE_N = pd.NamedAgg(column = 'V_u', aggfunc = 'count'),
                        TRANSFORMED_VALUE_CENTRAL_TENDENCY_C = pd.NamedAgg(column = 'V_c', aggfunc = most_freq_qualif),
                        TRANSFORMED_VALUE_DISPERSION = pd.NamedAgg(column = 'V_u', aggfunc = dispersion_agg)
                        )
        else:
            df_cur = df_cur.groupby(['original_ID', 'SMILES', 'MODULE', 'TRANSFORMATION'],
                        as_index = False, sort = False).agg(                    
                        TRANSFORMED_VALUE_CENTRAL_TENDENCY = pd.NamedAgg(column = 'V_u', aggfunc = aggregation_function),
                        TRANSFORMED_VALUE_N = pd.NamedAgg(column = 'V_u', aggfunc = 'count'),                        
                        TRANSFORMED_VALUE_DISPERSION = pd.NamedAgg(column = 'V_u', aggfunc = dispersion_agg)
                        )
            df_cur['TRANSFORMED_VALUE_CENTRAL_TENDENCY_C'] = '='
    else:
        print("Averaging valid TRANSFORMED_VALUE's after grouping by SMILES, MODULE, TRANSFORMATION.")
        print("NOTE: any instance of multiple distinct original_ID's per SMILES will be reported as a ';'-separated list.")
            # define a function that makes a semicolon-separated list of unique strings
        def cs_list_unique(s):
            s = np.unique(s) # pd.unique would preserve the order; but we want to sort
            if len(s) == 1 :
                return s[0]
            else:
                s = [str(si) for si in s]
                return ';'.join(s)
        if keep_only_numeric == False:
            df_cur = df_cur.groupby(['SMILES', 'MODULE', 'TRANSFORMATION'],
                        as_index = False, sort = False).agg(                    
                        original_ID = pd.NamedAgg(column = 'original_ID', aggfunc = cs_list_unique),
                        TRANSFORMED_VALUE_CENTRAL_TENDENCY = pd.NamedAgg(column = 'V_u', aggfunc = aggregation_function),
                        TRANSFORMED_VALUE_N = pd.NamedAgg(column = 'V_u', aggfunc = 'count'),
                        TRANSFORMED_VALUE_CENTRAL_TENDENCY_C = pd.NamedAgg(column = 'V_c', aggfunc = most_freq_qualif),
                        TRANSFORMED_VALUE_DISPERSION = pd.NamedAgg(column = 'V_u', aggfunc = dispersion_agg)
                        )
        else:
            df_cur = df_cur.groupby(['SMILES', 'MODULE', 'TRANSFORMATION'],
                        as_index = False, sort = False).agg(                    
                        original_ID = pd.NamedAgg(column = 'original_ID', aggfunc = cs_list_unique),
                        TRANSFORMED_VALUE_CENTRAL_TENDENCY = pd.NamedAgg(column = 'V_u', aggfunc = aggregation_function),
                        TRANSFORMED_VALUE_N = pd.NamedAgg(column = 'V_u', aggfunc = 'count'),                        
                        TRANSFORMED_VALUE_DISPERSION = pd.NamedAgg(column = 'V_u', aggfunc = dispersion_agg)
                        )
            df_cur['TRANSFORMED_VALUE_CENTRAL_TENDENCY_C'] = '='

    # Back-calculate VALUE_AVG by inverse transformation of TRANSFORMED_VALUE_CENTRAL_TENDENCY
    print("")
    print("Applying the inverse transformations to obtain VALUE_AVG.")
        
    df_cur['VALUE_AVG'] = [eval(t) if t != 'x' else x for x, t in \
        zip(df_cur['TRANSFORMED_VALUE_CENTRAL_TENDENCY'], df_cur['TRANSFORMATION'].map(transf_to_reverse_transf_functs))]

    # If required, keep the averaged uncensored value
    if keep_uncensored_and_qualifier_in_output == True :
        df_cur['TRANSFORMED_VALUE_CENTRAL_TENDENCY_U'] = df_cur['TRANSFORMED_VALUE_CENTRAL_TENDENCY'].copy()

    # Applying qualifiers to 'TRANSFORMED_VALUE_CENTRAL_TENDENCY' and 'VALUE_AVG'
    df_cur['TRANSFORMED_VALUE_CENTRAL_TENDENCY'] = [c + str(v) if c != '=' else v for c, v in \
        zip(df_cur['TRANSFORMED_VALUE_CENTRAL_TENDENCY_C'], df_cur['TRANSFORMED_VALUE_CENTRAL_TENDENCY'])]
    df_cur['VALUE_AVG'] = [c + str(v) if c != '=' else v for c, v in \
        zip(df_cur['TRANSFORMED_VALUE_CENTRAL_TENDENCY_C'], df_cur['VALUE_AVG'])]

    # Remove unnecessary columns
    if keep_uncensored_and_qualifier_in_output == True :
        df_cur.drop(columns = ['TRANSFORMED_VALUE', 'V_u', 'V_c'], inplace = True, errors = 'ignore')
    else:
        df_cur.drop(columns = ['TRANSFORMED_VALUE', 'TRANSFORMED_VALUE_CENTRAL_TENDENCY_C', 'V_u', 'V_c'], inplace = True, errors = 'ignore')

    # Rename the central tendency and dispersion columns according to what aggregation_function was
    df_cur.rename(columns = {
        'TRANSFORMED_VALUE_CENTRAL_TENDENCY' : TRANSFORMED_VALUE_CENTRAL_TENDENCY_colname,
        'TRANSFORMED_VALUE_CENTRAL_TENDENCY_C' : TRANSFORMED_VALUE_CENTRAL_TENDENCY_colname + '_C',
        'TRANSFORMED_VALUE_CENTRAL_TENDENCY_U' : TRANSFORMED_VALUE_CENTRAL_TENDENCY_colname + '_U',
        'TRANSFORMED_VALUE_DISPERSION' : TRANSFORMED_VALUE_DISPERSION_colname
        }, inplace = True, errors = 'ignore')

    # Discard records for which a MODULE has fewer than min_number_data_points    
    if min_number_data_points > 0 :
        print("")
        print("Looking for MODULE's with fewer than " +  str(min_number_data_points) + " data points.")
        df_counts = df_cur.groupby('MODULE')[TRANSFORMED_VALUE_CENTRAL_TENDENCY_colname].count()
        MODULES_to_drop = list(df_counts[df_counts < min_number_data_points].index)
        if len(MODULES_to_drop) > 0 :
            print("")
            for m, n in zip(MODULES_to_drop, df_counts.loc[MODULES_to_drop]) :
                print("Dropping '" + str(m) + "' (" + str(n) + " data points).")

            invalid_MODULES_index = df_cur[df_cur['MODULE'].isin(MODULES_to_drop)].index        
            df_invalid_MODULE, df_cur = df_cur.loc[invalid_MODULES_index], df_cur.drop(invalid_MODULES_index, axis = 0)

            if df_invalid_MODULE.shape[0] > 0 :
                print("")
                print("Saving averaged records of MODULE's with too few data points to : " + input_curated_unpivoted_csv_file_path + '_averged_too_few_data_points.csv.gz')
                df_invalid_MODULE.to_csv(input_curated_unpivoted_csv_file_path + '_too_few_data_points.csv.gz', index = False)

            if df_cur.shape[0] == 0:
                raise ValueError("After dropping the MODULE's with too few data points, nothing is left. Please review your parameters/data.")
        else:
            print("")
            print("No MODULE with too few data points found.")

    print("")
    print("Process completed. " + str(df_cur.shape[0]) + " final averaged records.")
    print("")
    print("Saving results to: " + output_averaged_unpivoted_csv_file_path)
    df_cur.to_csv(output_averaged_unpivoted_csv_file_path, index = False)

    
def F5_AUXFUN_write_csv_from_dataframe_with_sparse_cols(
    dataframe,
    sparse_columns_names,
    output_csv_file_full_path,
    ):
    """
    Writes a sparse pandas DataFrame to csv, saving a lot of time.
    (writing a sparse DataFrame to csv by standard pandas to_csv() for some reason is horribly slow)

    Parameters
    ----------
    dataframe : pandas.Dataframe
        - the DataFrame to process        
    sparse_column_names : list of strings
        - the names of the columns that are sparse in the input dataframe
        - NOTE: errors will be raised if this list includes dense columns
    output_csv_file_full_path : str
        - path to the csv file to write out
        - can end in .csv, .csv.gz, .csv.zip    

    Returns
    -------
    None
        Saving the resulting file is the only expected action by this function.
    """

    # remove output file if it exists, as it will be written in 'append' mode
    import os
    if os.path.exists(output_csv_file_full_path):
        os.remove(output_csv_file_full_path)

    # calculate how many chunks need to be made so that at each iteration ~10^8 cells are read
    N_chunks = np.ceil(dataframe.shape[0] * dataframe.shape[1] / 100000000).astype('int32')

    # the native pandas .to_csv function works well for this case, using mode = 'a';        
    # only, it needs a dense dataframe for good performance;
    # this is achieved by writing in chunks and densifying only each chunk, so not too much memory is used
    for i, chunk in enumerate(np.array_split(dataframe, N_chunks)):        
        chunk[sparse_columns_names] = chunk[sparse_columns_names].sparse.to_dense()            
        chunk.to_csv(output_csv_file_full_path, mode = 'a', header=i==0, index = False)
        
def F5_csv_unpivoted_std_avg_append_and_pivot(
    input_averaged_unpivoted_csv_file_path_list,
    output_pivoted_csv_file_path,
    central_tendency_colname = 'TRANSFORMED_VALUE_MEAN',
    use_transf_as_prefix = True,
    TRANSF_to_prefix_dictionary = {'x' : '', 'log10(x)' : 'log10_', 'log10(x/(1-x))' : 'logit_base10_', 'log10(x/(100-x))' : 'logit_base10_percent_'},
    use_prefix_per_input_file_on_original_ID = False,
    prefix_per_input_file = None,
    ):    
    """
    - Takes a list of paths to *unpivoted* *standardised* *averaged* files as input, as resulting from running F4_csv_unpivoted_std_transf_cur_to_averaged.
    - Standard columns are expected:
      ['original_ID', 'SMILES', 'MODULE', ('TRANSFORMATION'), central_tendency_colname]; any other column is ignored.
    - Appends all the files.
    - Aggregates records by the 'SMILES' column.
    - Pivots the data using the 'MODULE' column as new data column names, and the central_tendency_colname as new data column values.
    - Concatenates values of the 'original_ID' column by ';', optionally prefixing each by a specified string.
    - Saves the resulting pivoted file as a csv.
    
    Parameters
    ----------
    input_averaged_unpivoted_csv_file_path_list : list of string
        - paths to *unpivoted* *standardised* *averaged* files
        - use a list even if only one!
    output_pivoted_csv_file_path : str
        - path to the output csv file, can end in .csv, .csv.gz or .csv.zip
    central_tendency_colname : str
        - either 'TRANSFORMED_VALUE_MEAN' or 'TRANSFORMED_VALUE_MEDIAN'
        - IMPORTANT NOTE: must be the same in *all* input files!        
    use_transf_as_prefix : bool
        - defaults to True
        - uses the value of 'TRANSFORMATION' column to lool up the TRANSF_to_prefix_dictionary
          and prefix the resulting string to the new column name made from MODULE
        - example, if TRANSFORMATION == 'log10(x)', MODULE = 'rat_IV_CL' --> prefix = 'log10_'
          --> new column name = 'log10_rat_IV_CL'
        - this is advised to avoid misunderstandings on modelling output (is the prediction rat_CL or its log10?)
    TRANSF_to_prefix_dictionary : dict
         - applicable when use_transf_as_prefix == True
         - defaults to {'x' : '', 'log10(x)' : 'log10_', 'log10(x/(1-x))' : 'logit_base10_', 'log10(x/(100-x))' : 'logit_base10_percent_'}
    use_prefix_per_input_file_on_original_ID : bool
        - defaults to False
        - set to True if you are combining files from different sources and want to tag original_ID's accordingly
        - NOTE: this must mandatorily be set to True if the same MODULE name occurs in different files!
    prefix_per_input_file : None or list of strings
        - defaults to None
        - applicable when use_prefix_per_input_file_on_original_ID == True
        - if given, must be a list of the *same length* as input_averaged_unpivoted_csv_file_path_list
    
    Returns
    -------
    None
        Saving the resulting pivoted file is the only expected action by this function.
    """

    # v14: we suppress the option to merge by ID and SMILES; only CXSMILES is used for merging from now on    
    # rather that rewriting all the logic, we simply set to False the parameter that controlled this behaviour
    keep_distinct_original_ID_per_SMILES = False
        
    # Sanity check 1: is the input files parameter a list?
    if not isinstance(input_averaged_unpivoted_csv_file_path_list, list) :
        raise TypeError("input_averaged_unpivoted_csv_file_path_list is supposed to be a list of strings (even if it's only one string).")

    # Sanity check 2: if the prefix per file is given, is it a list, and of the correct length?
    if prefix_per_input_file != None:
        if not isinstance(prefix_per_input_file, list) :
            raise TypeError("prefix_per_input_file is supposed to be a list of strings (even if it's only one string).")
        if len(prefix_per_input_file) != len(input_averaged_unpivoted_csv_file_path_list):
            raise ValueError("prefix_per_input_file must have the same length as input_averaged_unpivoted_csv_file_path_list. (" \
                + str(len(prefix_per_input_file)) + ', ' + str(len(input_averaged_unpivoted_csv_file_path_list)) + ')')
    else:
        # And if it's not given, it is not possible to request to prefix original_ID's
        if use_prefix_per_input_file_on_original_ID == True :
            raise ValueError("To be able to add prefixes to original_ID's, you mus provide the list of strings 'prefix_per_input_file'.")

    # START PROCESSING

    df = pd.DataFrame()

    expected_unpivoted_cols = ['original_ID', 'SMILES', 'MODULE', central_tendency_colname]

    if use_transf_as_prefix == True:
        expected_unpivoted_cols.append('TRANSFORMATION')

    print("")
    print("Reading unpivoted averaged data...")
    print("")

    for i, f in enumerate(input_averaged_unpivoted_csv_file_path_list):
        dfi = pd.read_csv(f, usecols = None) # read all columns and later only keep the required ones        
        
        for c in expected_unpivoted_cols:
            if c not in dfi.columns :
                raise ValueError("Column '" + str(c) + "' is missing from file '" + str(f)+ "'.")
        
        # If it is required to use_transf_as_prefix, append it now to MODULE's
        # E.g. 'ASOL pH 7.4'  --> 'log10_ASOL pH 7.4'
        if use_transf_as_prefix == True:
            dfi['MODULE'] = [TRANSF_to_prefix_dictionary[t] + str(m) for t, m in zip(dfi['TRANSFORMATION'], dfi['MODULE'])]
        
        # If it is required to prefix_per_input_file, append it now to MODULE's
        # E.g. 'log10_ASOL pH 7.4' --> 'xxx_log10_ASOL pH 7.4'
        if prefix_per_input_file != None:
            dfi['MODULE'] = [str(prefix_per_input_file[i]) + '_' + str(m) for m in dfi['MODULE']]

        # If it is required to prefix original_ID's, do it now
        # E.g. 'G123454' --> 'XXX_G123454'
        if use_prefix_per_input_file_on_original_ID == True :
            dfi['original_ID'] = ['_'.join([str(prefix_per_input_file[i]), str(ID)]) for ID in dfi['original_ID']]
            
        df = pd.concat([df, dfi[expected_unpivoted_cols]], axis = 0, ignore_index = True)

    # Drop 'TRANSFORMATION': no longer needed
    df.drop(columns = ['TRANSFORMATION'], inplace = True, errors = 'ignore')

    # Sanity check 3: no multiple instances of central_tendency_colname must exist
    # for each group of {SMILES, MODULE} or {original_ID, SMILES, MODULE},
    # depending on the setting of keep_distinct_original_ID_per_SMILES
        
    if keep_distinct_original_ID_per_SMILES == True :
        groupby_list = ['original_ID', 'SMILES', 'MODULE']
    else:
        groupby_list = ['SMILES', 'MODULE']

    if any(df.groupby(groupby_list)[central_tendency_colname].count() > 1):
        raise ValueError("In at least one instance, grouping by {" + ','.join(groupby_list) + "} resulted in multiple values of " + central_tendency_colname + ".\nThis is not allowed, as it would require averaging, which is supposed to have been done at this stage.\nPlease review your input files.")
        
    print("Sorting and counting records by MODULE...")
    print("")

    df.sort_values(by = 'MODULE', inplace = True, ignore_index = True)
    sorted_prop_counts = df.groupby('MODULE')['MODULE'].count()

    # Make the 2 fixed columns of the pivoted output, using the groupby_list minus MODULE
    groupby_list.remove('MODULE')

    print("Collecting unique pivot identifiers (grouping by {" + ','.join(groupby_list) + "})...")    

    if keep_distinct_original_ID_per_SMILES == True :        
        df_piv = df[groupby_list].copy()
        df_piv.drop_duplicates(inplace = True)
    else:        
        print("NOTE: any instance of multiple distinct original_ID's per SMILES will be reported as a ';'-separated list.")
        # define a function that makes a semicolon-separated list of unique strings
        def cs_list_unique(s):
            # New: 2024-02-21 - if s is itself a list of ;-separated lists, which it could be as an effect of it coming from F4,
            # we need to sub-split it for np.unique to work correctly
            s_unsplit = []
            for si in s :
                s_unsplit.extend(str(si).split(';'))            
            s = np.unique(s_unsplit) # pd.unique would preserve the order; but we want to sort
            if len(s) == 1 :
                return s[0]
            else:                
                s = [str(si) for si in s]
                return ';'.join(s)
        df_piv = df.groupby(groupby_list,
                as_index = False, sort = False)['original_ID'].apply(cs_list_unique)
    
    # reset the index
    df_piv.reset_index(inplace = True, drop = True)

    # Create a temporary pivot_index in df_piv, a copy of its current index
    df_piv['pivot_index'] = list(df_piv.index)

    print("")
    print("Mapping unique pivot identifiers back to the original data...")
    print("")

    # Merge df_piv's pivot_index into df
    cols_to_keep_df_piv = groupby_list.copy()
    cols_to_keep_df_piv.append('pivot_index')
    df = df.merge(df_piv[cols_to_keep_df_piv], how = 'left', on = groupby_list)

    # make the first 2 columns of df_piv sparse, to preserve the sparsity on later merge
    df_piv['original_ID'] = pd.arrays.SparseArray(df_piv['original_ID'])    
    df_piv['SMILES'] = pd.arrays.SparseArray(df_piv['SMILES'])

    # drop pivot_index, no longer needed
    df_piv.drop(columns = ['pivot_index'], inplace = True)

    # pivot the data, one MODULE at a time, exploting the fact that df is sorted and its MODULE occurrences counted, 
    # and pivot_index knows where the records from df must go in df_piv

    print("Pivoting...")
    print("")

    current_start = 0
    for c, l in zip(sorted_prop_counts.index, sorted_prop_counts.values) :
        print("   -> property: '" + c + "' ( " + str(l) + " records )")
        # find the records of df where property c is contained, and put the records into temporary dfi, including pivot_index
        indices = range(current_start, (current_start + l))
        dfi = df.loc[indices, [central_tendency_colname, 'pivot_index']].copy()
        # pivot dfi    
        dfi.set_index('pivot_index', inplace = True)
        dfi.rename(columns = {central_tendency_colname : c}, inplace = True)
        # join this pivoted column into df_piv, by index    
        df_piv = df_piv.join(dfi, on = None, how = 'left', sort = False)
        # make the newly added column sparse
        df_piv[c] = pd.arrays.SparseArray(df_piv[c])
        current_start += l

    print("")
    print("Process completed. " + str(df_piv.shape[0]) + " final pivoted records.")
    print("")
    print("Saving results to: " + output_pivoted_csv_file_path)
    
    # save the pivoted file in chunks, using the previously defined function (all columns are sparse)
    F5_AUXFUN_write_csv_from_dataframe_with_sparse_cols(
        dataframe = df_piv,
        sparse_columns_names = list(df_piv.columns),
        output_csv_file_full_path = output_pivoted_csv_file_path)