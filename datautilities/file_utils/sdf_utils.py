from rdkit import Chem
import gzip

def read_sdf_file(sdf_file):
    """
    Read the sdf file and return a list of molecules.

    Parameters
    ----------
    sdf_file : str
        The path of the sdf file to be read.

    Returns
    -------
    list
        A list of RDKit's Mol objects read from the sdf file.
    """
    if sdf_file.endswith("gz"):
        supplier = Chem.ForwardSDMolSupplier(gzip.open(sdf_file, "rb"))
    else:
        supplier = Chem.SDMolSupplier(sdf_file)

    mols = [mol for mol in supplier if mol is not None]
    return mols



def extract_num_atoms_and_properties_from_SDF(SDF_in, properties):
    """
    Read the sdf file, extract num atoms and specified properties, and return them as a list and as a dictionary, respectively.
    Any invalid molecule (resulting in a 'None' mol) will be *skipped*.

    Parameters
    ----------
    SDF_in : str
        The path of the sdf file to be read.
    properties : list
        A list of property names to be extracted from the molecules.

    Returns
    -------
    list, dict
        list : a list of the number of atoms for each valid molecule in the SD file.
        dict : a dictionary where keys are the property names and values are lists of corresponding properties for each valid molecule in the SD file.
    """
    supplier = Chem.SDMolSupplier(SDF_in)
    num_atoms_list = []
    properties_dict = {prop: [] for prop in properties}
    
    for mol in supplier :
        if mol is not None :
            num_atoms_list.append(mol.GetNumAtoms())
            for prop in properties:
                if mol.HasProp(prop):
                    properties_dict[prop].append(mol.GetProp(prop))
                else:
                    properties_dict[prop].append(None)
            
    return num_atoms_list, properties_dict

