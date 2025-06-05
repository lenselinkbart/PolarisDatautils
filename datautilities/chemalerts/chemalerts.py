#!/usr/bin/env python3

import argparse
import gzip
import csv
from pickle import loads, dumps
from os import getenv
from os.path import basename,dirname,join
from rdkit import Chem
#from rdkit.Chem import PandasTools
from itertools import takewhile
import pandas as pd
import multiprocessing as mp
from time import time,sleep
from datetime import timedelta
import sys
import logging
logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] %(levelname)s: %(message)s')



def deltaTime(startTime):
    return f'{timedelta(seconds=round(time()-startTime))}'

def parse_args():
   parsera = argparse.ArgumentParser(description='Run the model workflow')
   parsera.add_argument("-i", help="the input .csv")
   parsera.add_argument("-o", help="opt iterations")
   global args
   args = parsera.parse_args()


def process_chemalerts(csvfile):
    chemalerts = pd.read_csv(str(csvfile), delimiter = "\t")
    chemalerts = chemalerts["SMARTS"].tolist()
    #print(chemalerts)
    return chemalerts


class ForwardSmilesSupplier:
    def openFile(self, filePath):
        if  filePath.endswith("gz"):
            logging.debug(f"Unzipping {filePath}.")
            return gzip.open(filePath, mode='rt')
        else:
            return open(filePath, mode='rt')

    def __init__(self, file, delimiter=',', 
                             smilesColumn=1, 
                             nameColumn=0, 
                             titleLine=True):
        if isinstance(file, str):
            handle = self.openFile(file)
        elif not file.closed:
            handle = file
        else:
            logging.error(f'Unable to open {file}')
        self.reader         = csv.reader(handle, delimiter=delimiter)
        if titleLine:
            header = next(self.reader)
            if isinstance(smilesColumn, str):
                smilesColumn = header.index(smilesColumn)
                logging.debug(f'Inferening Smiles columns ({smilesColumn}) based on header ({header})')
            if isinstance(nameColumn, str):
                logging.debug(f'Inferening name columns ({nameColumn}) based on header ({header}).')
                nameColumn = header.index(nameColumn)
                
        self.smilesColumn   = smilesColumn
        self.nameColumn     = nameColumn

    def __iter__(self):
        return self
    def __next__(self):
        line = next(self.reader)
        if line and (mol:=Chem.MolFromSmiles(line[self.smilesColumn])):
            mol.SetProp("_Name", line[self.nameColumn])
            return mol
        else:
            return None


class ChemAlerts:
    def __init__(self, chemAlertsFile=None, delimiter='\t', nameColumn=1, levelColumn=6, smartsColumn=2):
        if chemAlertsFile is None:
            chemAlertsFile = join(dirname(__file__), "chemalerts.tsv")
        self.readChemAlerts(chemAlertsFile)
        self.nJobs = int(getenv("SLURM_CPUS_PER_TASK")) if getenv("SLURM_CPUS_PER_TASK") else int(mp.cpu_count())

    def readChemAlerts(self, chemAlertsFile, delimiter='\t', nameColumn=1, levelColumn=6, smartsColumn=2):
        logging.info(f'Reading "{chemAlertsFile}" for chemalerts.')
        start = time()
        level1Alerts=[]
        level2Alerts=[]
        with open(chemAlertsFile, mode='r') as f:
            reader = csv.reader(f, delimiter=delimiter)
            for line in reader:
                mol = Chem.MolFromSmarts(line[smartsColumn])
                if mol:
                    if line[levelColumn] == "1":
                        level1Alerts.append((line[nameColumn], mol))
                    elif line[levelColumn] == "2":
                        level2Alerts.append((line[nameColumn], mol))
                    else:
                        logging.warning(f"Level {line[levelColumn]} of {line[nameColumn]}, not understood!")
                else:
                    logging.warning(f'SMARTS string of {line[nameColumn]}, not understood! "{line[smartsColumn]}"')
        self.level2Alerts=level2Alerts
        self.level1Alerts=level1Alerts

        logging.info(f"{len(self.level1Alerts)} level 1 alerts read.")
        logging.info(f"{len(self.level2Alerts)} level 2 alerts read.")
        
        
    
    def updateAlert(self, index, newAlertLevel, newAlertName):
        alertLevel = self.df.loc[index, "Level"]
        if alertLevel >= newAlertLevel:
            if alertLevel > newAlertLevel:
                self.df.loc[index, "Level"] = newAlertLevel
                self.df.loc[index, "ChemAlerts_reason"] = ""
        
            if newAlertLevel == 1:
                if not self.df.loc[index, "ChemAlerts_reason"]:
                    self.df.loc[index, "ChemAlerts_recommendation"]    = "discard"
                    self.df.loc[index, "ChemAlerts_reason"]            = f'1 or more alerts of level 1: {newAlertName}'
                else:
                    self.df.loc[index, "ChemAlerts_reason"] += f', {newAlertName}'
            elif newAlertLevel == 2:
                # Count the number of level 2 alerts already reported
                numberOfAlerts = self.df.loc[index, "ChemAlerts_reason"].count(",")
                if numberOfAlerts == 0:
                    self.df.loc[index, "ChemAlerts_recommendation"] = "keep"
                    self.df.loc[index, "ChemAlerts_reason"] = f'less than 4 alerts of level 2: {newAlertName}'
                elif numberOfAlerts > 3:
                    self.df.loc[index, "Level"]                    = 1.5
                    self.df.loc[index, "ChemAlerts_recommendation"]= "discard"
                    self.df.loc[index, "ChemAlerts_reason"]        = f'less than 4 alerts of level 2: {newAlertName}'
                else:
                    self.df.loc[index, "ChemAlerts_reason"] += f', {newAlertName}'
            else:
                self.df.loc[index, "ChemAlerts_recommendation"]    = "keep"
                self.df.loc[index, "ChemAlerts_reason"]            = f'no alert'

    def checkRDKitForAlerts(self, row):
        alertMol = row["alertMol"]
        if alertMol:
            try:
                if 'Molecule' in self.df.columns:
                    for index,pickledMol in self.df.Molecule.items():
                        mol = loads(pickledMol)
                        if mol.HasSubstructMatch(alertMol):
                            self.updateAlert(index, row["Level"], row["Name"])
                elif 'SMILES' in self.df.columns:
                    for index,smiles in self.df.SMILES.items():
                        if mol := Chem.MolFromSmiles(smiles):
                            if mol.HasSubstructMatch(alertMol):
                                self.updateAlert(index, row["Level"], row["Name"])
                
            except ValueError as e:
                print('Invalid query: {}, SMARTS: "{}"'.format(row["Name"], row["SMARTS"]))
    
    def checkForLevel2(self, compounds, alerts):
        flaggedCompounds=[]
        #for idx, alertMol in self.chemAlerts_df.loc[self.chemAlerts_df.Level==2,"alertMol"].items():
        for alertReason, alertMol in self.level2Alerts:
            for title,pickledMol in compounds.items():
                mol = loads(pickledMol)
                if mol.HasSubstructMatch(alertMol):
                    #alertReason = self.chemAlerts_df.at[idx, "Name"]
                    if title not in alerts:
                        alerts[title] = \
                            {
                                'level'          : 2,
                                'recommendation' : 'keep',
                                'reason'         : f'less than 4 alerts of level 2: {alertReason}'
                            }
                        flaggedCompounds.append(title)
                    else:
                        reason = alerts[title]['reason']
                        nAlerts    = reason.count(',')
                        if nAlerts >= 3:
                            # Replace the first occurence of "less", by "more" and append new reason
                            reasons = reason.replace('less', 'more', 1) + f', {alertReason}'
                            alerts[title] = \
                                {
                                    'level'          : 1.5,
                                    'recommendation' : 'discard',
                                    'reason'         : reasons
                                }
                        else:
                            temp = alerts[title]
                            temp['reason'] += f', {alertReason}'
                            alerts[title]=temp
        return flaggedCompounds

    def checkForLevel1(self, compounds, alerts):
        flaggedCompounds=[]
        #for idx, alertMol in self.chemAlerts_df.loc[self.chemAlerts_df.Level==1,"alertMol"].items():
        for alertReason, alertMol in self.level1Alerts:
            for title,pickledMol in compounds.items():
                mol = loads(pickledMol)
                if mol.HasSubstructMatch(alertMol):
                    #alertReason = self.chemAlerts_df.at[idx, "Name"]
                    if title not in alerts:
                        alerts[title] = \
                            {
                                'level'          : 1,
                                'recommendation' : 'discard',
                                'reason'         : f'1 or more alerts of level 1: {alertReason}'
                            }
                        flaggedCompounds.append(title)
                    else:
                        temp = alerts[title]
                        temp['reason'] += f', {alertReason}'
                        alerts[title]=temp
        return flaggedCompounds

    def checkRDKitForAlerts(self, compounds, alerts):
        
        flaggedCompounds = self.checkForLevel1(compounds, alerts)
        for flaggedCompound in flaggedCompounds:
            del compounds[flaggedCompound]
        flaggedCompounds = self.checkForLevel2(compounds, alerts)

    def readNmols(self, suppl, nMols):
        i=0
        compounds={}
        for mol in takewhile(lambda mol: i<nMols, suppl):
            if mol:
                title=mol.GetProp('_Name')
                if title in compounds:
                    j=1
                    while((title:=f'{title}_{j}') in compounds):
                        j+=1
                    compounds[title]=dumps(mol)
                else:
                    compounds[mol.GetProp('_Name')]=dumps(mol)
                i+=1
            else:
                logging.debug("Invalid compound!")
        return compounds
    
    def submitJob(self, compounds, alerts, jobs):
        first=True
        while len(jobs) >= self.nJobs-1:
            jobs = [job for job in jobs if job.is_alive()]
            if len(jobs) == self.nJobs:
                sleep(1)
                if first:
                    print('Waiting for jobs to finish', end='')
                    first=False
                print('.',end='')

        job = mp.Process(target=self.checkRDKitForAlerts, args=(compounds,alerts))
        jobs.append(job)
        job.start()
        logging.debug(f'Starting new job ({job.pid}).')
        
        return jobs

    def openCompoundsFile(self, filePath, 
                            sep=',', SMILEScolumn=1, titleColumn=0):
        if  filePath.endswith("gz"):
            logging.debug(f'Unzipping "{filePath}".')
            fileHandle=gzip.open(filePath, 'rt')
        else:
            fileHandle=filePath
        
        if filePath.endswith((".mae",".maegz")):
            logging.debug(f"Parsing file ({filePath}) as a maestro file.")
            suppl = Chem.MaeMolSupplier(fileHandle)
        elif(filePath.endswith((".sdf", ".sdfgz", ".sdf.gz"))):
            logging.debug(f"Parsing file ({filePath}) as an sdf file.")
            suppl = Chem.ForwardSDMolSupplier(fileHandle)
        else:
            logging.debug(f"Parsing file ({filePath}) as SMILES.")
            suppl = ForwardSmilesSupplier(fileHandle, delimiter=sep, 
                                                smilesColumn=SMILEScolumn, 
                                                nameColumn=titleColumn, 
                                                titleLine=1)
        return suppl,fileHandle

    def check_file(self, filePath, sep=",", 
                        SMILEScolumn="SMILES", titleColumn=None):
        self.sourceFile =   {
                                'filePath'      : filePath,
                                'sep'           : sep,
                                'SMILEScolumn'  : SMILEScolumn,
                                'titleColumn'   : titleColumn
                            }
        suppl, fileHandle = self.openCompoundsFile(filePath, sep, SMILEScolumn, titleColumn)
        
        self.checkDataset(suppl)
        
        if 'close' in dir(fileHandle):
            fileHandle.close()
    def check_df(self, df):
        if "SMILES" in df:
            smiles_col = "SMILES"
        else:
            return -1
    
        def smiles_to_mol(smiles_title_tuple):
            smiles,title = smiles_title_tuple
            if mol:=Chem.MolFromSmiles(smiles):
                mol.SetProp("_Name", str(title))
            return mol
        
        suppl = map(smiles_to_mol, zip(iter(df[smiles_col]), iter(df.index)))
        self.checkDataset(suppl)

        alerts_df       = pd.DataFrame.from_dict(self.alerts, orient="index")
        alerts_df.index = alerts_df.index.astype(df.index.dtype, copy=False)
        
        level_3     = {"level":3, "recommendation":"keep", "reason":"no alert"}
        level_3_idx = df.index[~df.index.isin(alerts_df.index)]
        level_3_df  = pd.DataFrame([level_3 for _ in range(len(level_3_idx))], index=level_3_idx)
        alerts_df   = pd.concat([alerts_df, level_3_df])

        df= df.join(alerts_df)
        return df

    def checkDataset(self, suppl):
        start = time()

        i=0
        with mp.Manager() as manager:
            
            logging.debug(f"Number of available cpus: {self.nJobs}")
            jobs = []
            alerts=manager.dict()
            
            #Run chemalerts on every {chunkSize} number of compounds
            chunkSize=int(1e5)
            while len(compounds:=self.readNmols(suppl, chunkSize)) >= chunkSize:
                jobs = self.submitJob(compounds, alerts, jobs)
                i+=chunkSize

            # Submit last compounds   
            i+=len(compounds)
            jobs = self.submitJob(compounds, alerts, jobs)
            
            logging.debug("Joining jobs.")
            for job in jobs:
                job.join()
            logging.debug("Jobs joined.")
            
            self.alerts = alerts.copy()
            
            level1Alerts  = 0
            level15Alerts = 0
            level2Alerts  = 0
            for alert in self.alerts.values():
                if alert['level'] == 1:
                    level1Alerts+=1
                elif alert['level'] == 1.5:
                    level15Alerts+=1
                elif alert['level'] == 2:
                    level2Alerts+=1
        if i==0:
            logging.error("No compounds processed. Did you provide a valid input file? ")

                
        logging.info(f'All ({i:,.0f}) molecules where processed ({time() - start:.3f}s).')
        logging.info(f'{level1Alerts:,.0f} level 1 warning found.')
        logging.info(f'{level15Alerts:,.0f} level 1.5 warning, {level2Alerts:,.0f} level 2 warnings found.')

     
        return i!=0

    def writeSafeUnSafeSMILES(self, safeFile, unsafeFile):
        if self.sourceFile["filePath"].endswith(".gz"):
            sourceFileHandle = gzip.open(self.sourceFile["filePath"], mode='rt')
            safeFileHandle   = gzip.open(safeFile,   mode='wt')
            unSafeFileHandle = gzip.open(unsafeFile, mode='wt')
        else:
            sourceFileHandle = open(self.sourceFile["filePath"], mode='rt')
            safeFileHandle   = open(safeFile,   mode='wt')
            unSafeFileHandle = open(unsafeFile, mode='wt')
        
        reader           = csv.reader(sourceFileHandle, delimiter=self.sourceFile["sep"])
        safeFileWriter   = csv.writer(safeFileHandle,   delimiter=self.sourceFile["sep"])
        unSafeFileWriter = csv.writer(unSafeFileHandle, delimiter=self.sourceFile["sep"])

        # Write file header
        header = next(reader)
        safeFileWriter.writerow(header)
        unSafeFileWriter.writerow(header)

        # Loop over the rest of the input file.
        # Write out safe compound to the safe file, 
        #   unsafe compounds will be written to the unsafe file
        titleColumn = self.sourceFile["titleColumn"]
        if isinstance(titleColumn, str):
            logging.debug(f'Inferening Smiles columns ({titleColumn}) based on header ({header})')
            titleColumn = header.index(titleColumn)
        alertTitles = self.alerts.keys()
        for line in reader:
            if line[titleColumn] not in alertTitles:
                # Safe compound
                safeFileWriter.writerow(line)
            else:
                # Unsafe compound
                unSafeFileWriter.writerow(line)
        
        # Close file handles
        sourceFileHandle.close
        safeFileHandle.close()
        unSafeFileHandle.close()

    def writeSafeUnSafeSDF(self, safeFile, unsafeFile):
        if self.sourceFile["filePath"].endswith(".gz"):
            sourceFileHandle = gzip.open(self.sourceFile["filePath"], mode='rt')
            safeFileHandle   = gzip.open(safeFile,   mode='wt')
            unSafeFileHandle = gzip.open(unsafeFile, mode='wt')
        else:
            sourceFileHandle = open(self.sourceFile["filePath"], mode='rt')
            safeFileHandle   = open(safeFile, mode='wt')
            unSafeFileHandle = open(safeFile, mode='wt')
        
    
        # Read the first title
        title=sourceFileHandle.readline()
        alertTitles = self.alerts.keys()
        safeCompound=(title not in alertTitles)
        while(line:=sourceFileHandle.readline()):
            if safeCompound:
                safeFileHandle.write(line)
            else:
                unSafeFileHandle.write(line)
            
            if line == '$$$$':
                title=sourceFileHandle.readline()
                if title not in alertTitles:
                    # Safe compound
                    safeCompound=True
                    safeFileHandle.write(title)
                else:
                    # Unsafe compound
                    safeCompound=False
                    unSafeFileHandle.write(title)
        sourceFileHandle.close
        safeFileHandle.close()
        unSafeFileHandle.close()


    def saveResult(self, warningsFile, safeFile=None, unsafeFile=None, delimiter=','):
        logging.info("Saving data...")
        startt = time()
        if warningsFile.endswith(".gz"):
            warningsFileHandle = gzip.open(warningsFile, mode='wt')
        else:
            warningsFileHandle = open(warningsFile, mode='wt')

        writer = csv.writer(warningsFileHandle, delimiter=delimiter)
        writer.writerow(["Title", "Level", "Recommendation", "Reason"])
        for key,value in self.alerts.items():
            writer.writerow([key, value["level"], value["recommendation"], value["reason"]])
        warningsFileHandle.close()

        if safeFile is None and unsafeFile is None:
            logging.info(f'Saved only the warning ({deltaTime(startt)} s).')
            return

        if self.sourceFile["filePath"].endswith((".sdf","sdf.gz")):
            self.writeSafeUnSafeSDF(safeFile, unsafeFile)
        else:
            self.writeSafeUnSafeSMILES(safeFile, unsafeFile)
        

        logging.info(f'Saved the results ({deltaTime(startt)}).')

def parseArguments(argv):
    parser = argparse.ArgumentParser(description="Scan compounds for chemical alerts.")
    parser.add_argument('SMILESfile', type=str,
                            help='The tab delimited file, that contains all the chemalert SMARTS.')
    parser.add_argument('SMARTSfile', type=str,
                            help='The tab delimited file, that contains all the chemalert SMARTS.')
    parser.add_argument('--resultFile', type=str, 
                            help='The result file in which the chemical alerts will be saved.')
    parser.add_argument('--safeFile', type=str, 
                            help='The result file containing the safe structures.')
    parser.add_argument('--unSafeFile', type=str, 
                            help='The result file containing the unsafe structures.')
    parser.add_argument('--SMILESsep', type=str, default=",",
                            help='The seperator (used for both reading SMILES and writing the results).')
    parser.add_argument('--SMARTSsep', type=str, default='\t',
                            help='The seperator when reading the SMARTS file.')
    parser.add_argument('--SMARTSlevelColumn', type=int, default=6,
                            help='The column containg the chemical alerts level (SMARTS).')
    parser.add_argument('--SMARTSnameColumn', type=int, default=1,
                            help='The column containg the chemical alert name (SMARTS).')
    parser.add_argument('--SMARTScolumn', type=int, default=2,
                            help='The column containing the SMARTS-strings (SMARTS).')
    parser.add_argument('--SMILEScolumn', default="smiles",
                            help='The column containing the smiles.')
    parser.add_argument('--titleColumn', default="Title",
                            help='The column containing the compound ID (title).')
                            
    parser.add_argument('--verbose', type=bool, default=False,
                            help='Enable debug output')          
    arguments = parser.parse_args()

    if not arguments.resultFile:
        resultPath = dirname(arguments.SMILESfile)
        resultFile = basename(arguments.SMILESfile)
        resultStem,resultExt = resultFile.split('.',  1)
        
        arguments.resultFile = \
            join(resultPath, f'{resultStem}_warnings.{resultExt}')
        if not arguments.safeFile:
            arguments.safeFile = \
                join(resultPath, f'{resultStem}_safe.{resultExt}')
        if not arguments.unSafeFile:
            arguments.unSafeFile = \
                join(resultPath, f'{resultStem}_unSafe.{resultExt}')

    return arguments

def defineLogger(arguments):
    rdkitLogger=logging.getLogger('Chem')
    
    for handle in rdkitLogger.handlers:
        handle.setFormatter(logging.getLogger().__format__)

def main(argv):

    # parse input arguments
    arguments = parseArguments(argv)
    defineLogger(arguments)

    startt = time()
    logging.info("Reading chemical alerts.")
    alertObj = ChemAlerts(arguments.SMARTSfile, delimiter    = arguments.SMARTSsep,
                                                levelColumn  = arguments.SMARTSlevelColumn, 
                                                nameColumn   = arguments.SMARTSnameColumn, 
                                                smartsColumn = arguments.SMARTScolumn)
    logging.debug(f'Initiating took {deltaTime(startt)}!')
    
    logging.info("Checking for chemical alerts.")
    startt = time()
    result = alertObj.checkDataset(arguments.SMILESfile, sep=arguments.SMILESsep, 
                                        titleColumn=arguments.titleColumn,
                                        SMILEScolumn=arguments.SMILEScolumn)
    logging.debug(f'Checking took {deltaTime(startt)}!')
    if result:
        logging.info(f"Saving results to {arguments.resultFile}.")
        start = time()
        alertObj.saveResult(arguments.resultFile, arguments.safeFile, arguments.unSafeFile, arguments.SMILESsep)
        logging.debug(f'Saving took {time() - start:.3f}s!')
    
    logging.info("Done")

if __name__ == '__main__':
    main(sys.argv)