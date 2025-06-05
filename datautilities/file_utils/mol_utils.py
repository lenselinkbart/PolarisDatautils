def extract_properties(mols, properties):
    """
    Extract specified properties from a list of molecules.

    Parameters
    ----------
    mols : list
        A list of RDKit's Mol objects.
    properties : list
        A list of property names to be extracted from the molecules.

    Returns
    -------
    dict
        A dictionary where keys are the property names and values are lists of corresponding properties for each molecule.
    """
    data = {prop: [] for prop in properties}
    for mol in mols:
        for prop in properties:
            if mol.HasProp(prop):
                data[prop].append(mol.GetProp(prop))
            else:
                data[prop].append(None)
    return data

def get_specific_properties(mols, properties):
    """
    Read the sdf file, extract specified properties, and return them as a dictionary.

    Parameters
    ----------
    sdf_file : str
        The path of the sdf file to be read.
    properties : list
        A list of property names to be extracted from the molecules.

    Returns
    -------
    dict
        A dictionary where keys are the property names and values are lists of corresponding properties for each molecule in the sdf file.
    """
    data = extract_properties(mols, properties)
    return data

def extract_num_atoms(mols):
    """
    Extract the number of atoms from a list of molecules.

    Parameters
    ----------
    mols : list
        A list of RDKit's Mol objects.

    Returns
    -------
    list
        A list of the number of atoms for each molecule.
    """
    return [mol.GetNumAtoms() for mol in mols]

