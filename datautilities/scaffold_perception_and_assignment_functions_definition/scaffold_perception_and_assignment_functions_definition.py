
"""Define functions to do scaffold perception and scaffold assignment

    Examples of this submodule can be found in the examples 
"""

# Import the required modules
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import rdScaffoldNetwork
from rdkit.Chem import rdMolDescriptors
import math

# this is necessary to disable errors and warnings from rdkit, which seem to slow down the process and occupy too much space
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# S0: rdkit-based scaffold perception and validation

def S0_scaffold_perception_and_validation(SMILES_list,
                                          min_frequency = 2,
                                          min_NRG_HAC_threshold = 14,
                                          min_NRG_NumRings_threshold = 4,
                                          min_HAC = 8,
                                          min_NumRings = 2,
                                         ):
    """
    Takes as input a list of SMILES and scaffold validation parameters.
    Creates a scaffold network *per molecule* usind rdkit's Chem.Scaffolds.rdScaffoldNetwork.
    Validates the scaffolds and returns a list with the results.
    
    Parameters
    ----------
    SMILES_list : list of strings
        the SMILES of the compounds for which scaffold perception is required
        - if any SMILES is not valid, it will be skipped;
          in any case, the results contain a list of valid indices, see Returns
    # Validation rules section: preliminary note on terminology
    - HAC : heavy atom count (how many non-H atoms the scaffold contains, not including R groups)
    - NRG : number of R groups ('*' characters in the scaffold SMILES)
    - NumRings : number of rings (NOT ring *assemblies*, so indole counts for 2 rings)
    - frequency : how many molecules contain the scaffold    
    min_frequency : integer >= 0
        the minimal number of compounds that must contain a scaffold, for it to be considered valid
        - important filter, gets rid of nonsensical scaffolds that are actually 'rare' substructures of molecules and 'muddy the waters' in further analysis
        - advised to set this parameter to at least 2
        - set to 1 to neutralise this rule
    min_NRG_HAC_threshold : integer >= 0
        parameter trying to make NRG and HAC 'compensate' one another;
        applies the rule NRG >= floor(min_NRG_HAC_threshold / HAC)
        - smaller scaffolds need to have more R groups to be considered valid
        - larger scaffolds can 'get away' with fewer R groups
        - e.g. for a value of 14:
            a scaffold with HAC = 3 - 4 needs to have 3 R groups
            a scaffold with HAC = 5 - 7 needs to have 2 R groups
            a scaffold with HAC = 8 - 14 needs to have 1 R group
            a scaffold with HAC >= 15 has no limitation on the NRG
        - set to 0 to neutralise this rule
    min_NRG_NumRings_threshold : integer >= 0
        parameter trying to make NRG and HAC 'compensate' one another;
        applies the rule NRG + NumRings >= min_NRG_NumRings_threshold
        - scaffolds with fewer rings need to have more R groups to be considered valid
        - complementary to min_NRG_HAC_threshold, which can be 'fooled' by single rings with heavy decoration;
          e.g. a Ph with a single R group and CF3 falls into the HAC 8-14 bracket and only needs 1 R group;
          adding this rule with value set to 3 removes the R-Ph-CF3 case (1 ring + 1 R group = 2 < 3)
        - set to 0 to neutralise this rule
    min_HAC : integer >= 0
        minimal HAC for a scaffold to be considered valid
        - not easy to identify a 'one size fits all' parameter
        - for some specific sets, really not definable a priori what the minimal HAC should be
        - advised to use 8
        - set to 0 to neutralise this rule
    min_NumRings : integer >= 0
        minimal NumRings for a scaffold to be considered valid
        - not easy to identify a 'one size fits all' parameter
        - for some specific sets, really not definable a priori what the minimal NumRings should be
        - advised to use 2
        - set to 0 to neutralise this rule
    
    Returns
    -------
    List of lists:
    0. indices of *valid* SMILES (the indices of the SMILES in the input list, which could be processed);
        NOTE: *all lists whose index refers to _molecules_ are synced with this one*
        - e.g. [0, 3, 4...] means that input SMILES_list[1] and SMILES_list[2] has to be skipped, so the
          molecule with index 1 in all lists refers to the SMILES at position 3 in the input list.
        - take this into account if/when mapping the results back to your original data
          (e.g. fill a list with NA's and replace its elements [0, 3, 4...] with the output of this function).
    1. list of lists of *validated* scaffolds SMILES of each valid molecule (passing the validation rules)
    2. list of lists of *rejected* scaffolds SMILES of each valid molecule (filtered out by the validation rules)
    3. list of *validated* scaffold SMILES, in the order in which they were indexed
        NOTE: *all lists whose index refers to _scaffolds_ are synced with this one*
        - e.g. the validated scaffold with index 2 has the SMILES at position 2 of this list
    4. list of lists of *validated* scaffold indices of each valid molecule       
        - e.g. [[0, 1, 3], [0, 2], ...] means that:
            the first valid molecule contains validated scaffolds 0, 1 and 3;
            the second valid molecule contains validated scaffolds 0 and 2; etc.
            (this is basically the index version of 1, for conveniency in further use, e.g. for assignment)
    5. lists of lists of indices of the molecules that contain each *validated* scaffold
        - e.g. [[11, 13, 18], [15, 16], ...] means that:
            the first validated scaffold was found in valid molecules 11, 13, 18;
            the second validated scaffold was found in valid molecules 15, 16; etc.
            (this is basically the inverse of 4)
        - by mapping 'len' on this list you obtain the *frequency* of each validated scaffold
    6. list of lists of the form [HAC, NRG, NumRings] of each *validated* scaffold
        - e.g. [[10, 2, 3], [6, 3, 1], ...] means that:
            the first validated scaffold had 10 heavy atoms, 2 R groups and 3 rings;
            the second validated scaffold had 6 heavy atoms, 3 R groups and 1 ring; etc.
            (this list can be of use for selecting of the 'best' scaffolds during scaffold assignment)
    7. list of integers
        'ranks' of the scaffolds, for use in assignment (everything else being equal, a scaffold with smaller rank is preferred)
        - currently based on decreasing HAC, NumRings, NRG;
          thus favouring larger scaffolds with more rings and more R groups
        - e.g. [4, 3, 5, 7, ...] meaning: if the scaffolds with index 0, 1, 2, 3 were equivalent in some evaluation,
          break their ties using ranks [4, 3, 5, 7], so in this case, select the scaffold with index 1 (rank 3)
        """
    # Set the scaffold enumeration parameters, first because the function does not run without them, second to obtain what we need vs the default output
    params = rdScaffoldNetwork.ScaffoldNetworkParams()
    params.flattenChirality = False
    params.includeGenericScaffolds = False
    params.pruneBeforeFragmenting = False
    params.includeScaffoldsWithoutAttachments = False

    # 0. Do the initial scaffold perception, without attempting any filtering for now    
    print('Automated scaffold perception by rdkit rdScaffoldNetwork')
    print('--------------------------------------------------------')

    # List of indices of the input SMILES for which the conversion to molecule and scaffold network calculation worked without errors
    mols_original_indices = []
    # Index of the latest perceived scaffold
    initial_sc_idx = -1
    # Dictionary linking the scaffold index each scaffold SMILES
    sc_idx_vs_sc_smiles_dict = dict()
    # List containing the perceived scaffold index for each valid mol, indexed as the range of mols_original_indices
    initial_sc_idx_vs_mol_idx_list = []
    # Dictionary listing [HAC, NRG, NumRings] for each scaffold index
    metrics_vs_sc_idx_dict = dict()
    # Dictionary listing frequency for each scaffold index
    frequency_vs_sc_idx_dict = dict()

    for sm_idx, sm in enumerate(SMILES_list) :
        try:
            m = Chem.MolFromSmiles(sm)
            if m != None :
                sn = rdScaffoldNetwork.CreateScaffoldNetwork([m], params)
                sn_smiles = list(sn.nodes)
                sn_sc_idx = []            
                for sn_sm in sn_smiles :
                    if sn_sm not in sc_idx_vs_sc_smiles_dict :
                        initial_sc_idx += 1
                        sc_index = initial_sc_idx
                        sc_idx_vs_sc_smiles_dict[sn_sm] = sc_index
                        sn_m = Chem.MolFromSmiles(sn_sm)
                        HAC = sn_m.GetNumHeavyAtoms()
                        NRG = sn_sm.count('*')
                        NumRings = rdMolDescriptors.CalcNumRings(sn_m)
                        metrics_vs_sc_idx_dict[sc_index] = [HAC, NRG, NumRings]
                        frequency_vs_sc_idx_dict[sc_index] = 1
                    else :
                        sc_index = sc_idx_vs_sc_smiles_dict[sn_sm]
                        frequency_vs_sc_idx_dict[sc_index] += 1
                    sn_sc_idx.append(sc_index)
                mols_original_indices.append(sm_idx)
                initial_sc_idx_vs_mol_idx_list.append(sn_sc_idx)            
        except:
            pass

    # convert each dict to temporary lists
    frequency_vs_initial_sc_idx_list = list(frequency_vs_sc_idx_dict.values())
    del frequency_vs_sc_idx_dict
    sc_smiles_vs_initial_sc_idx_list = list(sc_idx_vs_sc_smiles_dict.keys())
    del sc_idx_vs_sc_smiles_dict
    metrics_vs_initial_sc_idx_list = list(metrics_vs_sc_idx_dict.values())
    del metrics_vs_sc_idx_dict

    # Provide some information to the user
    print('\n' + str(len(SMILES_list)) + ' input SMILES evaluated.')
    print('> of which ' + str(len(initial_sc_idx_vs_mol_idx_list)) + ' were valid and could be processed.')
    print('> resulting in a total of ' + str(len(frequency_vs_initial_sc_idx_list)) + ' unique scaffolds.')

    # 1. Validate the perceived scaffolds
    # New compared to step 0: make a dictionary of molecule indices vs scaffold index (lists of molecules that contain each scaffold)
    print('\nStarting scaffold validation...')

    sc_idx_vs_initial_sc_idx_dict = dict() # mapping of initial sc_idx to new sc_idx
    mol_idx_vs_sc_idx_dict = dict()
    sc_idx_vs_mol_idx_list = []
    metrics_vs_sc_idx_list = []
    sc_idx = -1
    rejected_sc_smiles_vs_mol_idx = []

    for valid_mol_idx, initial_sc_idx in enumerate(initial_sc_idx_vs_mol_idx_list) :
        # for each valid molecule, go through its list of perceived scaffolds and validate them using the above defined rules    
        sn_sc_idx = []
        rejected_sc_smiles_vs_mol_idx.append([])
        initial_sc_idx_0 = initial_sc_idx[0]
        # Remove the parent molecule from the list to validate, unless there is only that
        if len(initial_sc_idx) > 1 :
            initial_sc_idx.pop(0)
        # Validation
        validated_initial_sc_idx = []
        for initial_sc_idxi in initial_sc_idx :
            initial_sc_idxi_metrics = metrics_vs_initial_sc_idx_list[initial_sc_idxi]
            HAC = initial_sc_idxi_metrics[0]
            NRG = initial_sc_idxi_metrics[1]
            NumRings = initial_sc_idxi_metrics[2]
            frequency = frequency_vs_initial_sc_idx_list[initial_sc_idxi]
            validation_bool = ((frequency >= min_frequency) and \
                               (NRG >= math.floor(min_NRG_HAC_threshold / HAC)) and \
                               (NRG + NumRings >= min_NRG_NumRings_threshold) and \
                               (HAC >= min_HAC) and \
                               (NumRings >= min_NumRings)
                              )
            if validation_bool == True :
                validated_initial_sc_idx.append(initial_sc_idxi)
            else :
                rejected_sc_smiles_vs_mol_idx[-1].append(sc_smiles_vs_initial_sc_idx_list[initial_sc_idxi])
        if len(validated_initial_sc_idx) == 0 :
            validated_initial_sc_idx = [initial_sc_idx_0]
        # Create or update the dictionary and list entries with the new scaffold indices and info
        for initial_sc_idxi in validated_initial_sc_idx :
            if initial_sc_idxi not in sc_idx_vs_initial_sc_idx_dict :
                sc_idx += 1
                sc_idx_vs_initial_sc_idx_dict[initial_sc_idxi] = sc_idx            
                sc_index = sc_idx
                mol_idx_vs_sc_idx_dict[sc_index] = [valid_mol_idx]
                metrics_vs_sc_idx_list.append(metrics_vs_initial_sc_idx_list[initial_sc_idxi])
            else :
                sc_index = sc_idx_vs_initial_sc_idx_dict[initial_sc_idxi]            
                mol_idx_vs_sc_idx_dict[sc_index].append(valid_mol_idx)
            sn_sc_idx.append(sc_index)
        sc_idx_vs_mol_idx_list.append(sn_sc_idx)

    # 2. Format conversions

    # Convert dicts into lists, for faster reference by index
    mol_idx_vs_sc_idx_list = list(mol_idx_vs_sc_idx_dict.values())
    del mol_idx_vs_sc_idx_dict
    sc_smiles_vs_sc_idx_list = [sc_smiles_vs_initial_sc_idx_list[initial_sc_idx] for initial_sc_idx in list(sc_idx_vs_initial_sc_idx_dict.keys())]
    del sc_smiles_vs_initial_sc_idx_list
    del sc_idx_vs_initial_sc_idx_dict
    del metrics_vs_initial_sc_idx_list

    # Create a list with the lists of validated scaffold SMILES
    validated_sc_smiles_vs_mol_idx = [[sc_smiles_vs_sc_idx_list[sc_idxi] for sc_idxi in sc_idxs] for sc_idxs in sc_idx_vs_mol_idx_list]
    
    # Create a ranking for the scaffolds based on HAC, NumRings, NRG
    metrics_vs_sc_idx_list_HAC_NRG_NumRings_tuple = tuple(zip(*metrics_vs_sc_idx_list))
    sorted_sc_idx = [i for _, _, _, i in sorted(zip(list(metrics_vs_sc_idx_list_HAC_NRG_NumRings_tuple[0]),
                                                 list(metrics_vs_sc_idx_list_HAC_NRG_NumRings_tuple[2]),
                                                 list(metrics_vs_sc_idx_list_HAC_NRG_NumRings_tuple[1]),
                                                 list(range(len(metrics_vs_sc_idx_list)))), reverse = True)]
    # sorted_sc_idx is the list of sc_idx in the order that favours larger scaffolds with more rings and more R groups
    # To turn this into a *rank* of each sc_idx, we just repeat the sorting using indices:
    order_sc_idx = [i for _, i in sorted(zip(sorted_sc_idx, list(range(len(sorted_sc_idx)))))]

    # Provide some information to the user
    print('\n> remaining : ' + str(len(mol_idx_vs_sc_idx_list)) + ' validated unique scaffolds.')
    print('\nPerception and validation completed.')
    
    # Return the results
    return [
        mols_original_indices,        
        validated_sc_smiles_vs_mol_idx,
        rejected_sc_smiles_vs_mol_idx,
        sc_smiles_vs_sc_idx_list,
        sc_idx_vs_mol_idx_list,
        mol_idx_vs_sc_idx_list,
        metrics_vs_sc_idx_list,
        order_sc_idx
    ]

# S1: Shannon Entropy-guided scaffold assignment

def S1_Shannon_Entropy_guided_scaffold_assignment(sc_ids_vs_mol_list,
                                                  order_sc_idx,
):
    """
    Takes as input scaffold perception information for a set of molecules (see Parameters for details).
    Assigns a single 'preferred' scaffold to each molecule, using a method based on Shannon Entropy.
    
    Parameters
    ----------
    sc_ids_vs_mol_list : list of lists of integers or strings
        NOTE: this is the *only* mandatory parameter
        lists of *unique* scaffold 'identifiers' for each molecule in the set (can also be SMILES)
        - e.g. [[0, 1, 3], [0, 2], ...] means that:
            the first molecule contains scaffolds 0, 1 and 3;
            the second molecule contains scaffolds 0 and 2; etc.
        - e.g. [['x', 'a', 'fd'], ['yz', 'h2'], ...] means that:
            the first molecule contains scaffolds 'x', 'a' and 'fd';
            the second molecule contains scaffolds 'yz' and 'h2'; etc.
        - example of invalid entry: ['x', 'y', 'x', 'a'] raises an error, because 'x' is repeated
        - this list has length N, where N is the number of molecules in the set
    order_sc_idx : list of lists of integers (or None)
        lists of *ranks* of the scaffolds
        - if given, must have length M, where M is the overall number of *unique* identifiers in sc_ids_vs_mol_list
        - IMPORTANT: the order of the ranks must be the same as the *order of appearance* of *unique* identifiers
            e.g. if sc_ids_vs_mol_list is [[0, 1, 3], [0, 2], ...] and order_sc_idx is [12, 2, 3, 0, ...],
            these are the ranks of the scaffolds with id [0, 1, 3, 2, ...] respectively
            NOTE: if you use as sc_ids_vs_mol_list the 'list of lists of *validated* scaffold indices' from
            function S0_scaffold_perception_and_validation, the order of appearance is [0, 1, 2, ..., M - 1]
        - for use in assignment (everything else being equal, a scaffold with smaller rank is preferred)
        - e.g. [4, 3, 5, 7, ...] meaning: if the scaffolds with index 0, 1, 2 were equivalent in some evaluation,
          break their ties using ranks [4, 3, 5], so in this case, select the scaffold with index 1 (rank 3)
        
    Returns
    -------
    List of lists:
    0. list of integers (starting at 0) of length equal to len(sc_ids_vs_mol_list)
        the index of the cluster each molecule belongs to
        NOTE: the max of this list + 1 is the total number of clusters made
    1. list of sc_id's of length equal to len(sc_ids_vs_mol_list)
        the sc_id of the assigned scaffold for each molecule in the set
        - may be useful when the sc_id's are SMILES, to have a chemically meaningful cluster representation
    """
    print('Automated scaffold assignment guided by Shannon Entropy')
    print('-------------------------------------------------------')
    # Count the initial N (total number of molecules)
    N = len(sc_ids_vs_mol_list)

    # Create matching dicts for the user sc_id vs progressive sc_idx
    # and convert sc_id_vs_mol to sc_idx_vs_mol_idx_list.
    # At the same time, detect duplicates and halt with an error if any exists.
    sc_idx_vs_sc_id_dict = dict()
    sc_idx = -1
    sc_idx_vs_mol_idx_list = []
    for mol_idx, sc_ids in enumerate(sc_ids_vs_mol_list) :
        sc_idx_vs_mol_idxi = []
        sc_id_temp_dict = dict()
        for sc_id in sc_ids :
            # check for duplicates
            if sc_id not in sc_id_temp_dict :
                sc_id_temp_dict[sc_id] = 1
            else :
                print('\Error: duplicated scaffold id (' + str(sc_id) + ') found in molecule ' + str(mol_idx) + '. Please review your input.')
                return None
            if sc_id not in sc_idx_vs_sc_id_dict :
                sc_idx += 1
                sc_idx_vs_sc_id_dict[sc_id] = sc_idx
                sc_idxi = sc_idx
            else :
                sc_idxi = sc_idx_vs_sc_id_dict[sc_id]
            sc_idx_vs_mol_idxi.append(sc_idxi)
        sc_idx_vs_mol_idx_list.append(sc_idx_vs_mol_idxi)
    
    # Create a mapping from sc_idx to sc_id by inverting sc_idx_vs_sc_id_dict
    sc_id_vs_sc_idx = list(sc_idx_vs_sc_id_dict.keys())
    del sc_idx_vs_sc_id_dict

    # Count the number of unique scaffolds
    M = sc_idx + 1

    print('\n' + str(N) + ' molecules with ' + str(M) + ' unique scaffolds initially present in the set.')

    # Create the mol_idx_vs_sc_idx_list by inversion of sc_idx_vs_mol_idx_list
    mol_idx_vs_sc_idx_list = [[] for _ in range(M)]
    for mol_idx, sc_idxs in enumerate(sc_idx_vs_mol_idx_list) :
        for sc_idx in sc_idxs :    
            mol_idx_vs_sc_idx_list[sc_idx].append(mol_idx)
            
    # 3. Calculation of the initial Shannon-Entropy-related parameters (all mols in a single set)
    # Make a list of scaffold frequencies vs scaffold index (in how many molecules is this scaffold contained?)
    sc_freq_vs_sc_idx_list = [len(ml) for ml in mol_idx_vs_sc_idx_list]

    # Make a list of frequencies of frequencies (from 0 to N)
    freq_of_freqs_list = [0] * (N + 1)
    for sc_freq in sc_freq_vs_sc_idx_list :
        freq_of_freqs_list[sc_freq] += 1

    # Calculate SE and SEN
    Ui = [0] + [x * math.log2(x) for x in range(1, N + 1)] # convenience list with pre-calculated values of SE contribution per frequency
    #SE_ni = sum(Ui[ni] for ni in sc_freq_vs_sc_idx_list if ni >= 2)
    SE_ni = sum(frf * Ui[fr] for fr, frf in enumerate(freq_of_freqs_list) if ((fr >= 2) and (frf != 0)))
    #SE_Nni = sum(Ui[N - ni] for ni in sc_freq_vs_sc_idx_list if N-ni >= 2)
    SE_Nni = sum(frf * Ui[N - fr] for fr, frf in enumerate(freq_of_freqs_list) if ((N - fr >= 2) and (frf != 0)))
    SE = M * math.log2(N) - (SE_ni + SE_Nni) / N
    SEN = SE / M

    # 4. Iterative splitting

    # When a scaffold sc_idx is selected, all the N_sc mols that contain it (from mol_idx_vs_sc_idx_list) must migrate to a new set.
    # This causes the following changes in the *original* set:
    # - N drops by N_sc
    # - the frequencies (in sc_freq_vs_sc_idx_list) of all the sc_idx contained in the moved N_sc mols (from sc_idx_vs_mol_idx_list) drop according to their counts
    # - M drops by the number of sc_idx that go from > 0 to 0 in sc_freq_vs_sc_idx_list
    # In the *new* set instead:
    # - N_new_set = N_sc
    # - the frequencies (in sc_freq_list_new_set) of all the sc_idx contained in the moved N_sc mols (from sc_idx_vs_mol_idx_list) increase (from 0) according to their counts
    # - M_new_set = number of unique sc_idx in the moved mols

    # Create a dictionary of the sc_idx currently present in the set
    current_sc_idx_dict = dict.fromkeys(list(range(M)), [])

    # Initialise the clusters list (index = mol_idx)
    clusters = [0 for _ in range(N)]
    # Initialise the sc_id_vs_cluster_index_dict
    sc_id_vs_cluster_index_dict = dict()
    # Initialise the cluster index
    max_cl_idx = -1
    # Initialise a dataframe for the clustering output
    #df_clustering = pd.DataFrame()

    # 4.1. First, identify singletons (molecules whose scaffolds *all* have frequency 1, i.e. they are all associated only to that molecule)
    # --> these molecules can only form clusters of size 1; just as well to remove them upfront
    print('\n1. Detecting possible singletons...\n')

    for mol_idx, sc_idxs in enumerate(sc_idx_vs_mol_idx_list) :
        all_1 = True
        for sc_idx in sc_idxs :
            if sc_freq_vs_sc_idx_list[sc_idx] != 1 :
                all_1 = False
                break
        if all_1 == True :
            max_cl_idx += 1
            print('Molecule ' + str(mol_idx) + ' is a singleton : assigned to cluster ' + str(max_cl_idx))
            clusters[mol_idx] = max_cl_idx
            # in this case the sc_id of the cluster cannot be determined; should be the whole molecule; use its first scaffold
            sc_id_vs_cluster_index_dict[max_cl_idx] = sc_ids_vs_mol_list[mol_idx][0]
            N -= 1
            M -= len(sc_idxs)
            #df_new_cluster = pd.DataFrame({
            #    'mol_idx' : mol_idx,
            #    'cluster' : max_cl_idx,
            #    'cluster_scaffold_SMILES' : SMILES_list[mol_idx],
            #    'cluster_N_mols' : 1,
            #    'cluster_N_unique_scaffolds' : len(sc_idxs),
            #    'cluster_overlap' : 0
            #}, index = [0])
            #df_clustering = pd.concat([df_clustering, df_new_cluster], ignore_index = True)
            # Clean up the list of current available sc_idx, update the frequency of frequencies list (only element 1 changes) and the main frequency list
            for sc_idx in sc_idxs :
                del current_sc_idx_dict[sc_idx]
                freq_of_freqs_list[1] -= 1
                sc_freq_vs_sc_idx_list[sc_idx] = 0

    print('\nDone. Remaining molecules = ' + str(N) + ' with ' + str(M) + ' unique scaffolds.')

    # Update SE and SEN
    if N > 0 :
        #SE_ni = sum(Ui[sc_freq_vs_sc_idx_list[sc_idx]] for sc_idx in current_sc_idx_dict.keys() if sc_freq_vs_sc_idx_list[sc_idx] >= 2)
        SE_ni = sum(frf * Ui[fr] for fr, frf in enumerate(freq_of_freqs_list) if ((fr >= 2) and (frf != 0)))
        #SE_Nni = sum(Ui[N - sc_freq_vs_sc_idx_list[sc_idx]] for sc_idx in current_sc_idx_dict.keys() if N-sc_freq_vs_sc_idx_list[sc_idx] >= 2)
        SE_Nni = sum(frf * Ui[N - fr] for fr, frf in enumerate(freq_of_freqs_list) if ((N - fr >= 2) and (frf != 0)))
        SE = M * math.log2(N) - (SE_ni + SE_Nni) / N
        SEN = SE / M

    # 4.2. Do the assignment for non-singleton cases

    print('\n2. Assigning non-singleton molecules to scaffolds...')

    while len(current_sc_idx_dict) > 0 :

        # Calculate the change in N, M, SE, SEN that would occur in the old and new set by moving the mols with each of the sc_idx to the new set,
        # plus metrics that help with the selection of which sc_idx to choose, and store them as lists in the above dict
        # keys: available sc_idx
        # values: [N_new_set, M_new_set, SE_new_set, SEN_new_set, N_leftover, M_leftover, SE_leftover, SEN_leftover, SE_split, SEN_split, overlap, freq_new_set_vs_sc_idx, real_mol_idx_to_move]
        # Also keep lists SE_split_list and overlap_list, to allow making the choice of the sc_idx to use for splitting, at each step
        SE_split_list = []
        overlap_list = []
        for sc_idx_to_try in list(current_sc_idx_dict.keys()) :
            mol_idx_to_move = mol_idx_vs_sc_idx_list[sc_idx_to_try]
            real_mol_idx_to_move = [] # this is needed because we do not remove mol_idx from each element of mol_idx_vs_sc_idx_list, but we set to [] in sc_idx_vs_mol_idx_list when a mol is removed; see below
            freq_new_set_vs_sc_idx = dict()
            for mol_idx in mol_idx_to_move :
                sc_idx_in_mol_idx = sc_idx_vs_mol_idx_list[mol_idx]
                # any mol with no sc_idx was actually already removed
                if len(sc_idx_in_mol_idx) != 0 :
                    real_mol_idx_to_move.append(mol_idx)
                    for sc_idx in sc_idx_in_mol_idx :
                        if sc_idx not in freq_new_set_vs_sc_idx :
                            freq_new_set_vs_sc_idx[sc_idx] = 1
                        else :
                            freq_new_set_vs_sc_idx[sc_idx] += 1
            N_new_set = len(real_mol_idx_to_move)
            M_new_set = len(freq_new_set_vs_sc_idx)
            sc_freq_list_new_set = list(freq_new_set_vs_sc_idx.values())
            SE_ni_new_set = sum(Ui[ni] for ni in sc_freq_list_new_set if ni >= 2)
            SE_Nni_new_set = sum(Ui[N_new_set - ni] for ni in sc_freq_list_new_set if N_new_set-ni >= 2)
            SE_new_set = M_new_set * math.log2(N_new_set) - (SE_ni_new_set + SE_Nni_new_set) / N_new_set
            SEN_new_set = SE_new_set / M_new_set
            N_leftover = N - N_new_set

            # For now, we skip the 'more efficient' calculation (see section wrapped in a 'if False' statement) and use the full list of frequencies left over in the old set
            sc_freq_vs_sc_idx_list_leftover = sc_freq_vs_sc_idx_list.copy()
            for sc_idxi, sc_freqi in list(freq_new_set_vs_sc_idx.items()) :
                sc_freq_vs_sc_idx_list_leftover[sc_idxi] -= sc_freqi    
            SE_ni_leftover = sum(Ui[ni] for ni in sc_freq_vs_sc_idx_list_leftover if ni >= 2)
            #print('SE_ni_leftover calculated by all bits = ' + str(SE_ni_leftover))
            #SE_Nni_leftover_terms = [str(Ui[N_leftover - ni]) for ni in sc_freq_vs_sc_idx_list_leftover if ni > 0]
            #print(' + '.join(SE_Nni_leftover_terms))
            # NOTE: we skip the bits that are now 0; that allows us to ignore them in M as well
            SE_Nni_leftover = sum(Ui[N_leftover - ni] for ni in sc_freq_vs_sc_idx_list_leftover if ((ni >= 1) and (N_leftover - ni >= 2)))
            #print('SE_Nni_leftover calculated by all bits = ' + str(SE_Nni_leftover))
            M_leftover = sum(1 for sc_freq in sc_freq_vs_sc_idx_list_leftover if sc_freq > 0)

            # NOTE to self: the mistake is not to consider that when N goes to N_leftover, all the (N-ni)*log2(N-ni) terms change, in principle.
            # Solution: use a frequency of frequencies table, like in previous work.
            if False :
                # For SE_leftover, instead of calculating a modified sc_freq_vs_sc_idx_list and repeating the whole sum, we can just subtract the removed terms and add in the new ones
                #removed_sc_idx = []
                N_removed_sc_idx = 0
                SE_ni_change = 0
                SE_Nni_change = 0
                for sc_idxi, sc_freqi in list(freq_new_set_vs_sc_idx.items()) :
                    freq_current = sc_freq_vs_sc_idx_list[sc_idxi]
                    freq_goes_to = freq_current - sc_freqi
                    print('Scaffold ' + str(sc_idxi) + ' goes from frequency ' +  str(freq_current) + ' to ' + str(freq_goes_to))
                    if freq_goes_to == 0 :
                        #removed_sc_idx.append(sc_idxi)
                        N_removed_sc_idx += 1
                    else :                
                        # We must only add this term when freq_goes_to != 0, so M_leftover can be used below
                        SE_Nni_change += Ui[N_leftover - freq_goes_to]
                        print('Adding to SE_Nni_leftover the contribution from scaffold ' + str(sc_idxi) + ' (' +  str(Ui[N_leftover - freq_goes_to]) + ')')
                    SE_ni_change += Ui[freq_goes_to]
                    SE_ni_change -= Ui[freq_current]
                    SE_Nni_change -= Ui[N - freq_current]
                    print('Removing from SE_Nni_leftover the contribution from scaffold ' + str(sc_idxi) + ' (' +  str(Ui[N - freq_current]) + ')')
                M_leftover = M - N_removed_sc_idx
                SE_ni_leftover = SE_ni + SE_ni_change
                print('SE_ni_leftover calculated by reduced method = ' + str(SE_ni_leftover))
                SE_Nni_leftover = SE_Nni + SE_Nni_change
                print('SE_Nni_leftover calculated by reduced method = ' + str(SE_Nni_leftover))        

            if N_leftover > 0 :
                SE_leftover = M_leftover * math.log2(N_leftover) - (SE_ni_leftover + SE_Nni_leftover) / N_leftover
                SEN_leftover = SE_leftover / M_leftover        
            else :
                SE_leftover = 0
                SEN_leftover = 0

            SE_split = (N_new_set *  SE_new_set + N_leftover * SE_leftover) / N
            SEN_split = SEN_new_set + SEN_leftover
            overlap = M_new_set + M_leftover - M

            SE_split_list.append(SE_split)
            overlap_list.append(overlap)

            current_sc_idx_dict[sc_idx_to_try] = [N_new_set, M_new_set, SE_new_set, SEN_new_set, N_leftover, M_leftover, SE_leftover, SEN_leftover, SE_split, SEN_split, overlap, freq_new_set_vs_sc_idx, real_mol_idx_to_move]

        # Select the sc_idx to use for the split based on SE_split and overlap
        sc_idx_left = list(current_sc_idx_dict.keys())
        order_sc_idx_left = [order_sc_idx[sc_idxi] for sc_idxi in sc_idx_left]    
        selected_sc_idx = sorted(zip(SE_split_list, overlap_list, order_sc_idx, sc_idx_left))[0][3]
        data_for_sc_idx = current_sc_idx_dict[selected_sc_idx]
        SE_split = data_for_sc_idx[8]
        overlap = data_for_sc_idx[10]
        N_new_set = data_for_sc_idx[0]
        M_new_set = data_for_sc_idx[1]
        N_leftover = data_for_sc_idx[4]
        M_leftover = data_for_sc_idx[5]
        freq_new_set_vs_sc_idx = data_for_sc_idx[11]
        SE_leftover = data_for_sc_idx[6]
        SEN_leftover = data_for_sc_idx[7]
        real_mol_idx_to_move = data_for_sc_idx[12]

        print('\nScaffold # ' + str(selected_sc_idx) + ' was selected based on minimal SE_split (' + '{:.3f}'.format(SE_split) + ') and minimal overlap (' + str(overlap) + ').')
        max_cl_idx += 1
        print('Cluster # ' + str(max_cl_idx) + ' formed, with ' + str(N_new_set) + ' molecule(s) removed from the main set.')
        print(str(M - M_leftover) + ' unique scaffold(s) removed from main set.')
        print('Remaining = ' + str(N_leftover) + ' molecule(s) with ' + str(M_leftover) + ' unique scaffold(s).' )
        for mol_idx in real_mol_idx_to_move :
            clusters[mol_idx] = max_cl_idx
            sc_idx_vs_mol_idx_list[mol_idx] = [] # this is done to avoid having to suppress all moved mols in all relevant elements of mol_idx_vs_sc_idx_list
        sc_id_vs_cluster_index_dict[max_cl_idx] = sc_id_vs_sc_idx[selected_sc_idx]
        N = N_leftover
        M = M_leftover
        # Update the cluster report
        #df_new_cluster = pd.DataFrame({
        #    'mol_idx' : real_mol_idx_to_move,
        #    'cluster' : max_cl_idx,
        #    'cluster_scaffold_SMILES' : sc_id_vs_sc_idx[selected_sc_idx],
        #    'cluster_N_mols' : N_new_set,
        #    'cluster_N_unique_scaffolds' : M_new_set,
        #    'cluster_overlap' : overlap
        #}, index = list(range(len(real_mol_idx_to_move))))
        #df_clustering = pd.concat([df_clustering, df_new_cluster], ignore_index = True)
        # Update the frequencies lists
        for sc_idx, freq_new_set in list(freq_new_set_vs_sc_idx.items()) :        
            freq_current = sc_freq_vs_sc_idx_list[sc_idx]
            freq_goes_to = freq_current - freq_new_set
            sc_freq_vs_sc_idx_list[sc_idx] = freq_goes_to
            # When freq_goes_to is 0, the sc_idx is no longer available
            if freq_goes_to == 0 :
                del current_sc_idx_dict[sc_idx]
        # Update SE and SEN (although not strictly necessary in this implementation)
        SE = SE_leftover
        SEN = SEN_leftover

    # Map back the cluster indices to their original sc_id
    unique_sc_id_vs_mol = [sc_id_vs_cluster_index_dict[cl_idx] for cl_idx in clusters]

    return [clusters, unique_sc_id_vs_mol]