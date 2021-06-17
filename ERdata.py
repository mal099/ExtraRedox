import math
import numpy as np
from collections import Counter
import argparse
import os
import datetime

# Argument parser
classifier_parser = argparse.ArgumentParser(description='Translate data into a form understandable by ML algorithms')

classifier_parser.add_argument('-i', '--input',
                      metavar='input name',
                      action='store',
                      type=str,
                      help='Name of input file')
classifier_parser.add_argument('-o', '--output',
                      metavar='output path',
                      action='store',
                      type=str,
                      help='Path to output files')

classifier_parser.add_argument('-aa',
                      '--aminos',
                      action='store_true',
                      help='use amino acid feature')
classifier_parser.add_argument('-ss',
                      '--secstruc',
                      action='store_true',
                      help='use secondary structure feature')
classifier_parser.add_argument('-hse',
                      '--exposure',
                      action='store_true',
                      help='use half sphere exposure feature')
classifier_parser.add_argument('-acc',
                      '--accessibility',
                      action='store_true',
                      help='use accessibility feature')
classifier_parser.add_argument('-pka',
                      '--dissociation',
                      action='store_true',
                      help='use acid dissociation constant feature')
classifier_parser.add_argument('-ptm',
                      '--modifications',
                      action='store_true',
                      help='use PTMs feature')
classifier_parser.add_argument('-con',
                      '--concavity',
                      action='store_true',
                      help='use concavity feature')
classifier_parser.add_argument('-sta',
                      '--statistics',
                      action='store_true',
                      help='use statistics based AA and secstruc features')

classifier_parser.add_argument('-num',
                      '--numAAs',
                      action='store',
                      type=int,
                      choices=range(1, 20),
                      help='determines how many AAs are considered')
classifier_parser.add_argument('-amod',
                      '--AAmode',
                      action='store',
                      type=int,
                      choices=[0, 1],
                      help='determines if AA data is summed up (0) or treated separately (1)') #check if mode 1 is even still valid
classifier_parser.add_argument('-smod',
                      '--SSmode',
                      action='store',
                      type=int,
                      choices=[0, 1],
                      help='determines if SS data is summed up (0) or treated separately (1)')
classifier_parser.add_argument('-seq',
                      '--sequence',
                      action='store_true',
                      help='use this if the features were collected in sequence mode')
classifier_parser.add_argument('-imp',
                      '--impute',
                      action='store_true',
                      help='determines if imputation is used')

classifier_parser.add_argument('-all',
                      '--allfeat',
                      action='store_true',
                      help='use all features')
classifier_parser.add_argument('-nost',
                      '--nostatistics',
                      action='store_true',
                      help='use all features except stats')

args = classifier_parser.parse_args()
input_path = args.input
output_path = args.output
if not input_path:
    input_path = 'feat_output'
if not output_path:
    output_path = 'data_output'

numberAAs = args.numAAs
if not numberAAs:
    if args.sequence:
        numberAAs = 20
    else:
        numberAAs = 13

mode = args.AAmode
SSmode = args.SSmode

if not os.path.exists('../data/feat/' + output_path):
    os.mkdir('../data/feat/' + output_path)
if not os.path.exists('../data/feat/' + output_path):
    os.mkdir('../data/feat/' + output_path)

inFile = open('../data/feat/' + input_path + '/feat.txt')
outFile = open('../data/feat/' + output_path + '/data.txt', 'w')

impute_bool = args.impute
AAswitch = args.aminos
SSswitch = args.secstruc
HSEswitch = args.exposure
accswitch = args.accessibility
pkaswitch = args.dissociation
phosphoswitch = args.modifications
concavswitch = args.concavity
statsswitch = args.statistics
seq = args.sequence
impute_bool = args.impute

if args.allfeat or args.nostatistics:
    AAswitch = True
    SSswitch = True
    HSEswitch = True
    accswitch = True
    pkaswitch = True
    phosphoswitch = True
    concavswitch = True
if args.allfeat:
    statsswitch = True

# Set up dictionaries for amino acids and secondary structures
AAdict =   {'A': [0.39, 0.39, 0.45, 0.52, 0.47, 0.65, 0.89, 0.39, 0.46, 0, 1, 0, 0, 1, 1, 0, 0],
            'C': [0.55, 0.48, 0.53, 0.34, 0.39, 1.00, 0.42, 0.75, 0.31, 1, 1, 0, 0, 1, 1, 0, 0],
            'U': [0.55, 0.48, 0.53, 0.34, 0.39, 1.00, 0.42, 0.75, 0.31, 1, 1, 0, 0, 1, 1, 0, 0],
            'D': [0.62, 0.49, 0.59, 0.87, 0.28, 0.17, 0.62, 0.21, 0.70, 1, 0, 1, 0, 0, 1, 0, 0],
            'E': [0.69, 0.61, 0.75, 1.00, 0.08, 0.07, 1.00, 0.28, 0.57, 1, 0, 1, 0, 0, 0, 0, 0],
            'F': [0.79, 0.83, 0.82, 0.45, 0.44, 0.78, 0.73, 0.71, 0.33, 0, 1, 0, 0, 0, 0, 1, 0],
            'G': [0.31, 0.26, 0.29, 0.55, 0.36, 0.67, 0.27, 0.31, 1.00, 0, 1, 0, 0, 1, 1, 0, 0],
            'H': [0.74, 0.67, 0.76, 0.71, 0.42, 0.35, 0.66, 0.43, 0.46, 1, 1, 1, 1, 0, 0, 0, 0],
            'I': [0.61, 0.73, 0.69, 0.42, 0.39, 0.87, 0.69, 0.89, 0.27, 0, 1, 0, 0, 0, 0, 1, 1],
            'K': [0.69, 0.74, 0.78, 1.00, 0.14, 0.04, 0.77, 0.37, 0.60, 1, 1, 1, 1, 0, 0, 0, 0],
            'L': [0.61, 0.73, 0.67, 0.44, 0.28, 0.91, 0.84, 0.65, 0.32, 0, 1, 0, 0, 0, 0, 0, 1],
            'M': [0.61, 0.72, 0.73, 0.47, 1.00, 0.37, 0.82, 0.61, 0.29, 0, 1, 0, 0, 0, 0, 0, 0],
            'N': [0.61, 0.50, 0.63, 0.88, 0.22, 0.19, 0.48, 0.26, 0.76, 1, 0, 0, 0, 0, 1, 0, 0],
            'P': [0.52, 0.49, 0.57, 0.84, 0.25, 0.24, 0.21, 0.17, 0.75, 0, 0, 0, 0, 0, 1, 0, 0],
            'Q': [0.69, 0.63, 0.71, 0.87, 0.25, 0.19, 0.80, 0.52, 0.47, 1, 0, 0, 0, 0, 0, 0, 0],
            'R': [0.84, 0.76, 0.88, 0.90, 0.31, 0.09, 0.76, 0.45, 0.51, 1, 0, 1, 1, 0, 0, 0, 0],
            'S': [0.47, 0.39, 0.45, 0.75, 0.28, 0.37, 0.36, 0.51, 0.69, 1, 0, 0, 0, 1, 1, 0, 0],
            'T': [0.54, 0.54, 0.55, 0.76, 0.36, 0.30, 0.48, 0.63, 0.51, 1, 0, 0, 0, 1, 1, 0, 0],
            'V': [0.53, 0.61, 0.61, 0.43, 0.28, 0.93, 0.57, 1.00, 0.23, 0, 1, 0, 0, 0, 1, 0, 1],
            'W': [1.00, 1.00, 1.00, 0.53, 0.19, 0.81, 0.64, 0.72, 0.37, 1, 1, 0, 0, 0, 0, 1, 0],
            'Y': [0.88, 0.85, 0.90, 0.72, 0.36, 0.37, 0.47, 0.78, 0.43, 1, 1, 0, 0, 0, 0, 1, 0],
            '-': [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0, 0, 0, 0, 0, 0, 0, 0],
            'X': [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0, 0, 0, 0, 0, 0, 0, 0]}

featList = ['AA_Mass', 'AA_Vol', 'AA_Area', 'AA_SEA1', 'AA_SEA2', 'AA_SEA3', 'AA_Alpha', 'AA_Beta', 'AA_turn', 'AA_Polar', 'AA_Non-Polar', 'AA_Charge', 'AA_Positive', 'AA_Tiny', 'AA_Small', 'AA_Aromatic', 'AA_Aliphatic']

SSdict = {'e': [1, 0],
          'h': [0, 1],
          ' ': [0, 0],
          's': [0, 0],
          't': [0, 0],
          'b': [1, 0],
          'g': [0, 1],
          '-': [0, 0],
          'i': [0, 1],
          'C': [0, 0],
          'E': [1, 0],
          'H': [0, 1]}

print ('...calculating...')
# Set things up
now = datetime.datetime.now()
outFile.write(now.strftime('# /%Y-%m-%d_%H-%M\n'))
numberFeat = len(AAdict['A'])
avg_pKa = 9.60623430962

# Go through every line in the file to translate data into a format readable by the SVM
for line in inFile:
# Here, we're setting up the bins for averaging all close AAs
    features = []
    if mode == 0:
        for i in range(numberFeat):
            features.append(0)
# First we get the data for the cysteine
    line = line.rstrip().split('$')[1]
    if len(line)>100:
        outFile.write(line[0])
# Secondary Structure
        if SSswitch == True:
            for val in range(len(SSdict['h'])):
                outFile.write(' Cys_SS' + str(val) + ':' + str(SSdict[line[2]][val]))
# ConCavity value
        if concavswitch == True:
            if  line[4:].split(' ')[0] != 'np.nan':
                conc = float(line[4:].split(' ')[0])
                outFile.write(' Cys_conc:' + str(float(line[4:].split(' ')[0])))
            else:
                outFile.write(' Cys_conc:np.nan')
# pKA value
        if pkaswitch == True:
            if line[4:].split(' ')[0] != '99.99' and line[4:].split(' ')[1] != 'N/A' and line[4:].split(' ')[1] != 'np.nan':
                pka = float(line[4:].split(' ')[1])
                outFile.write(' Cys_PKA:' + str(float(line[4:].split(' ')[1])/avg_pKa))
            else:
                if impute_bool == True and seq == True:
                    outFile.write(' Cys_PKA:np.nan') # No value still gets average
                else:
                    outFile.write(' Cys_PKA:1.0') # No value still gets average
            if line[4:].split(' ')[1] != '99.99' and line[4:].split(' ')[2] != 'N/A' and line[4:].split(' ')[2] != 'np.nan':
                pka = float(line[4:].split(' ')[2])
                outFile.write(' Cys_PKA:' + str(float(line[4:].split(' ')[2])/avg_pKa))
            else:
                if impute_bool == True:
                    outFile.write(' Cys_PKA:np.nan') # No value still gets average
                else:
                    outFile.write(' Cys_PKA:1.0') # No value still gets average
# HSE (half sphere exposure)
        if HSEswitch == True:
            exposure = line.split('(')[1].split(')')[0].split(',')
            try:
                outFile.write(' Cys_HSE1:' + str(float(exposure[0])/20.0))
                outFile.write(' Cys_HSE2:' + str(float(exposure[1][1:])/20.0))
            except:
                if impute_bool == True:
                    outFile.write(' Cys_HSE1:np.nan')
                    outFile.write(' Cys_HSE2:np.nan')
                else:
                    outFile.write(' Cys_HSE1:10')
                    outFile.write(' Cys_HSE2:10')
# DSSP Accesibility
        if accswitch == True:
            acces = line.split(')')[1].split(' ')[1].strip()
            outFile.write(' Cys_Acc:' + acces)
# Post translational mods
        if phosphoswitch == True:
            PTMs = line.split('!')[0].split(')')[1].split(' ')[-3:]
            outFile.write(' Phospho:' + PTMs[0])
            outFile.write(' Acetyl:' + PTMs[1])
            outFile.write(' Ubiquitin:' + PTMs[2])
# Now we get the data for all other AAs
        line = line.split('!')[1:]
        expoList1 = []
        expoList2 = []
        accesList = []
        SSList = []
        if SSmode == 0:
            SSwrite = []
            for SSint in range(len(SSdict['h'])):
                SSwrite.append(0)
        statisticsList = [0,0,0,0,0]
        sequenceList = []
# Set up lists with all the values for AAs, HSE, SS
        for j in range(numberAAs):
            AAvals = line[j].split('&')
            SSList.append(AAvals[0])
            exposure = AAvals[2].split('(')[1].split(')')[0].split(',')
            expoList1.append(exposure[0])
            expoList2.append(exposure[1][1:])
            accesList.append(AAvals[3])
            sequenceList.append(AAvals[1])
            if mode == 0:
                for AA in range(numberFeat):
                    features[AA] += AAdict[AAvals[1]][AA]
            elif mode == 1:
                for AA in range(numberFeat):
                    features.append(AAdict[AAvals[1]][AA])
# Add up the special statistics features
        if statsswitch == True and seq == True:
            if sequenceList[7] == 'C':
                statisticsList[0] += 1.0
            if sequenceList[12] == 'C':
                statisticsList[0] += 1.0
            if sequenceList[15] == 'C':
                statisticsList[0] += 1.0
            if sequenceList[9] == 'L':
                statisticsList[0] += 1.0
            if sequenceList[13] == 'G':
                statisticsList[0] += 1.0
            if sequenceList[2] == 'Q':
                statisticsList[0] += 1.0
            if sequenceList[1] == 'R' or sequenceList[1] == 'K':
                statisticsList[1] += 1.0
            if sequenceList[3] == 'R' or sequenceList[1] == 'K':
                statisticsList[1] += 1.0
            if sequenceList[5] == 'R' or sequenceList[1] == 'K':
                statisticsList[1] += 1.0
            if sequenceList[14] == 'R' or sequenceList[1] == 'K':
                statisticsList[1] += 1.0
            if sequenceList[15] == 'R' or sequenceList[1] == 'K':
                statisticsList[1] += 1.0
            if sequenceList[16] == 'R' or sequenceList[1] == 'K':
                statisticsList[1] += 1.0
            if sequenceList[10] == 'P' or sequenceList[1] == 'K':
                statisticsList[1] += 1.0
            statisticsList[2] = SSList[:10].count('e')
            statisticsList[3] = SSList[10:].count('h')
            statisticsList[4] = SSList.count(' ') + SSList.count('s') + SSList.count('e')
# Write the values into our file while making sure the values are in the same range
        feature_index = 0
        if AAswitch == True:
            for val in features:
                if feature_index >= numberFeat:
                    feature_index = 0
                if mode == 1:
                    outFile.write(' ' + featList[feature_index] + ':' + str(val))
                else:
                    outFile.write(' ' + featList[feature_index] + ':' + str(val/5.0))
                feature_index += 1
        if SSswitch == True:
            if SSmode == 1:
                for SS in SSList:
                    feature_index = 0
                    for j in range(len(SSdict['h'])):
                        outFile.write(' SecStruc' + str(feature_index) + ':' + str(SSdict[SS][j]))
                        feature_index += 1
            elif SSmode == 0:
                for SS in SSList:
                    feature_index = 0
                    for j in range(len(SSdict['h'])):
                        SSwrite[j] += SSdict[SS][j]
                for j in SSwrite:
                    outFile.write(' SecStruc' + str(feature_index) + ':' + str(j))
                    feature_index += 1
        if HSEswitch == True:
            feature_index = 0
            for expo in range(len(expoList1)):
                try:
                    outFile.write(' HSE1_' + str(feature_index) + ':' + str(float(expoList1[expo])/20.0))
                    outFile.write(' HSE2_' + str(feature_index) + ':' + str(float(expoList2[expo])/20.0))
                except:
                    outFile.write(' HSE1_' + str(feature_index) + ':np.nan')
                    outFile.write(' HSE2_' + str(feature_index) + ':np.nan')
                feature_index += 1
        if accswitch == True:
            feature_index = 0
            for acc in range(len(accesList)):
                outFile.write(' Accessibility' + str(feature_index) + ':' + accesList[acc])
                feature_index += 1
        if statsswitch == True:
            feature_index = 0
            for val in statisticsList:
                outFile.write(' Stats' + str(feature_index) + ':' + str(val))
                feature_index += 1
        outFile.write('\n')

print('Done!')
