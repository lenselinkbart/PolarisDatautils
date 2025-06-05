
import pandas as pd
from rdkit import Chem
from os import chdir
from tempfile import TemporaryDirectory
from unittest import TestCase, skip
from parameterized import parameterized

from pathlib import Path
TEST_DATA_DIR = Path(__file__).parents[0] / 'data'

from datautilities.data_curation_functions import (F1_SDF_to_csv_with_SMILES,
                                                   F2_AUXFUN_read_csv_to_dataframe_with_all_sparse_cols,
                                                   F3_csv_unpivoted_to_standard_transformed_curated,
                                                   F4_csv_unpivoted_std_transf_cur_to_averaged,
                                                   F5_AUXFUN_write_csv_from_dataframe_with_sparse_cols,
                                                   F5_csv_unpivoted_std_avg_append_and_pivot)

class DataCuration(TestCase):
    @staticmethod
    def read_sdf(fileName):
        if (filePath := Path(fileName)).exists() and filePath.stat().st_size != 0:
            suppl = Chem.SDMolSupplier(filePath.absolute().as_posix())
            mols = [mol for mol in suppl if mol]
            return mols
        else:
            return []
    
    @parameterized.expand([
        (
            TEST_DATA_DIR / "test_cases_for_rdkit_stereo_standardisation.sdf",
            TEST_DATA_DIR / "test_cases_for_rdkit_stereo_standardisation_F1_F3_pass.csv",
            TEST_DATA_DIR / "test_cases_for_rdkit_stereo_standardisation_F1_failed.sdf"
        ),
    ])
    @skip("Skipping test due to RDKit version issues")
    def test_F1_SDF_to_csv_with_SMILES(self, source, ref_passed, ref_failed):
        with TemporaryDirectory() as tmpdirname:
            chdir(tmpdirname)
            test_passed = "test_cases_for_rdkit_stereo_standardisation_F1_pass.csv"
            test_failed = "test_cases_for_rdkit_stereo_standardisation_F1_failed.sdf"
            
            # Standardize example sdf
            F1_SDF_to_csv_with_SMILES(source.as_posix(),
                                    pass_output_csv_file_path=test_passed,
                                    fail_output_SD_file_path=test_failed)
            # Read result csv's
            test_df = pd.read_csv(test_passed, index_col="Index")
            ref_df  = pd.read_csv(ref_passed,  index_col="Index")
            
            for idx,test_row in test_df.iterrows():
                ref_row = ref_df.loc[idx]
                assert ref_row.Expected_CXSMILES_after_running_F1 == test_row.SMILES, \
                        ('Processed SMILES not equal to reference record.\n' 
                        f'Test:"{test_row.SMILES}" != ref:"{ref_row.Expected_CXSMILES_after_running_F1}" (idx: {idx})\n'
                        f'Test row:\n{test_row.to_string()}\n\nVs. reference row:\n{ref_row.to_string()}\n')
                
            # Check failed molecules
            test_failed_mols = self.read_sdf(test_failed)
            ref_failed_mols  = self.read_sdf(ref_failed)
            def report_differences(ref, test):
                return f'{len(ref):,.0f} reference failed records vs {len(test):,.0f} processed failed records'
            assert len(test_failed_mols) == len(ref_failed_mols), report_differences()
                    
            assert all([a == b for a, b in zip(test_failed_mols, ref_failed_mols)])