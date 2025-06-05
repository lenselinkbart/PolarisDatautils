from unittest import TestCase
import pandas as pd
from pandas.testing import assert_frame_equal
from datautilities.chemalerts import ChemAlerts

ref_data = {
    "CATNO":["X1", "X2", "X3", "X4", "X5"],
    "SMILES": [
        "CCOc1ccc([C@H]2Cn3nnc(C(=O)Nc4ccc5c(c4)c4ccccc4n5CC)c3CO2)cc1Cl",
        "CCc1c(-c2noc(-c3cnn(-c4ccc(Br)cc4)c3-n3cccc3)n2)nnn1-c1ccccc1",
        "COc1cc(OC)c(-n2nnc(-c3nc(-c4cn(-c5ccc(F)cc5)nn4)no3)c2C)cc1Cl",
        "CCOC(=O)C(C)C=O",
        "CCOC(=O)CC(=O)C1=CC=NC=C1"
    ],
    "level": [
        2,2,3,1,2
    ]
}

class test_ChemAlerts(TestCase):
    
    def test_chemalerts(self):
        o  = ChemAlerts()
        df = pd.DataFrame(ref_data)
        df.set_index("CATNO", inplace=True)
        
        alerts_df = o.check_df(df[["SMILES"]])
        assert_frame_equal(alerts_df[["level"]], df[["level"]])
        