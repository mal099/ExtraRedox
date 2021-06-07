# Input: redoxDB file
# Output: Data file for machine learning with proteins being labeled as either containing (1) or not containing (0) redox-sensitive cysteines
# Features are auto-covariance and CTD, according to You et al. (2013), https://doi.org/10.1186/1471-2105-14-S8-S10

import math
import os
import random
import gc

# +++ Setup lists and vars +++
rdbFile = open('../data/redoxDB_clustered.txt')
outFile = open('../data/redoxprot_data.txt', 'w')
sequenceList = []
IDList = []
SSList = []
ASAList = []

# Find Auto Covariance, local descriptor
# Composition, Transition and Distribution (CTD) of amino acid attributes such as physicochemical properties, secondary structure, and solvent accessibility

lag = 4
autoCov = []
autoCov_ASA = []
CTD = []
CTD_SS = []
cys_list = []
feat_AA = []
feat_ASA = []
feat_CTD = []
feat_CTD_SS = []
taxList = []
lenList = []

groupDict = {'A': '0',
             'G': '0',
             'V': '0',
             'C': '1',
             'U': '1',
             'D': '2',
             'E': '2',
             'F': '3',
             'I': '3',
             'L': '3',
             'P': '3',
             'H': '4',
             'N': '4',
             'Q': '4',
             'W': '4',
             'K': '5',
             'R': '5',
             'M': '6',
             'S': '6',
             'T': '6',
             'Y': '6',
             'X': '3'}

SSDict = {'C': '0',
          'E': '1',
          'H': '2'}

'''groupDict = {'A': [1,0,0,0,0,0,0],
              'G': [1,0,0,0,0,0,0],
              'V': [1,0,0,0,0,0,0],
              'C': [0,1,0,0,0,0,0],
              'D': [0,0,1,0,0,0,0],
              'E': [0,0,1,0,0,0,0],
              'F': [0,0,0,1,0,0,0],
              'I': [0,0,0,1,0,0,0],
              'L': [0,0,0,1,0,0,0],
              'P': [0,0,0,1,0,0,0],
              'H': [0,0,0,0,1,0,0],
              'N': [0,0,0,0,1,0,0],
              'Q': [0,0,0,0,1,0,0],
              'W': [0,0,0,0,1,0,0],
              'K': [0,0,0,0,0,1,0],
              'R': [0,0,0,0,0,1,0],
              'M': [0,0,0,0,0,0,1],
              'S': [0,0,0,0,0,0,1],
              'T': [0,0,0,0,0,0,1],
              'Y': [0,0,0,0,0,0,1]}'''

AAdict =   {'A': [0.39, 0.52, 0.65, 0.89, 0.39, 0.46, 0, 1, 0, 0, 1, 0],
            'C': [0.48, 0.34, 1.00, 0.42, 0.75, 0.31, 1, 1, 0, 0, 1, 0],
            'U': [0.48, 0.34, 1.00, 0.42, 0.75, 0.31, 1, 1, 0, 0, 1, 0],
            'D': [0.49, 0.87, 0.17, 0.62, 0.21, 0.70, 1, 0, 1, 0, 1, 0],
            'E': [0.61, 1.00, 0.07, 1.00, 0.28, 0.57, 1, 0, 1, 0, 0, 0],
            'F': [0.83, 0.45, 0.78, 0.73, 0.71, 0.33, 0, 1, 0, 0, 0, 0],
            'G': [0.26, 0.55, 0.67, 0.27, 0.31, 1.00, 0, 1, 0, 0, 1, 0],
            'H': [0.67, 0.71, 0.35, 0.66, 0.43, 0.46, 1, 1, 1, 1, 0, 0],
            'I': [0.73, 0.42, 0.87, 0.69, 0.89, 0.27, 0, 1, 0, 0, 0, 1],
            'K': [0.74, 1.00, 0.04, 0.77, 0.37, 0.60, 1, 1, 1, 1, 0, 0],
            'L': [0.73, 0.44, 0.91, 0.84, 0.65, 0.32, 0, 1, 0, 0, 0, 1],
            'M': [0.72, 0.47, 0.37, 0.82, 0.61, 0.29, 0, 1, 0, 0, 0, 0],
            'N': [0.50, 0.88, 0.19, 0.48, 0.26, 0.76, 1, 0, 0, 0, 1, 0],
            'P': [0.49, 0.84, 0.24, 0.21, 0.17, 0.75, 0, 0, 0, 0, 1, 0],
            'Q': [0.63, 0.87, 0.19, 0.80, 0.52, 0.47, 1, 0, 0, 0, 0, 0],
            'R': [0.76, 0.90, 0.09, 0.76, 0.45, 0.51, 1, 0, 1, 1, 0, 0],
            'S': [0.39, 0.75, 0.37, 0.36, 0.51, 0.69, 1, 0, 0, 0, 1, 0],
            'T': [0.54, 0.76, 0.30, 0.48, 0.63, 0.51, 1, 0, 0, 0, 1, 0],
            'V': [0.61, 0.43, 0.93, 0.57, 1.00, 0.23, 0, 1, 0, 0, 1, 1],
            'W': [1.00, 0.53, 0.81, 0.64, 0.72, 0.37, 1, 1, 0, 0, 0, 0],
            'Y': [0.85, 0.72, 0.37, 0.47, 0.78, 0.43, 1, 1, 0, 0, 0, 0],
            '-': [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0, 0, 0, 0, 0, 0],
            'X': [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0, 0, 0, 0, 0, 0]}
'''
0.8688897141377957), 7
0.8097910914674437), 12
0.440537403291667), 11
0.40069524955189273), 9
0.03859497539413075), 0
0.00024800033267351196), 10
0.00017205687726883567), 2
6.806317226356312e-05), 5
3.357946987382046e-06), 4
3.125848166246039e-06), 8
2.556771924206481e-06), 6
2.0556020974187213e-07), 3
1.1011091113244044e-07), 1
'''
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

# +++ Gather data +++
#Gather sequences and identifiers
seqBool = False
IDBool = False
cys_bool = False
for line in rdbFile:
    if line[0:3] == 'ID:':
            ID = line.split(' ')[1].strip(' ')
            IDBool = True
    elif line[0:8] == 'GENENAME' and IDBool == True:
        cys_bool = True
    elif line[0:8] == 'TAXONOMY' and IDBool == True:
        tax = line.split(':')[1]
    elif line[0] == '>' and IDBool == True:
        if (cys_bool == True and random.random() > 0.0) or cys_bool == False:
            try:
                if os.stat('../data/psipred_files/' + ID + '.ss').st_size == 0 or os.stat('../data/ASAquick/asaq.' + ID + '.fasta/asaq.pred').st_size == 0:
                    print(ID + ' is empty.')
                    continue
                SSFile = open('../data/psipred_files/' + ID + '.ss')
                ASAFile = open('../data/ASAquick/asaq.' + ID + '.fasta/asaq.pred')
                seqBool = True
                IDList.append(ID)
                SSList.append('')
                taxList.append(tax)
                for line in SSFile:
                    SSList[-1] += line[7]
                ASAList.append([])
                for line in ASAFile:
                    ASAList[-1].append(line.split(' ')[2])
                if cys_bool == True:
                    cys_list.append(1)
                else:
                    cys_list.append(0)
                print(ID + ' found!')
            except:
                print(ID + ' not found!')
        cys_bool = False
        IDBool = False
    elif seqBool == True:
        seqBool = False
        sequenceList.append(line.rstrip('\n'))
        lenList.append(str(len(line)))

# +++ Auto Covariance +++
#Auto Covariance for AA
#12 Features
for sequence in sequenceList:
    for property in range(len(AAdict['A'])):
        for i in range(lag):
            autoCov.append(0)
            feat_AA.append('autoCovAA' + str(property) + '_'+ str(i))
            meanProperty = 0.0
            for AApos in range(len(sequence)):
                meanProperty += AAdict[sequence[AApos]][property]
            meanProperty = meanProperty/len(sequence)
            for AApos in range(len(sequence)-i):
                autoCov[-1] += (AAdict[sequence[AApos]][property] - meanProperty) * (AAdict[sequence[AApos + i]][property] - meanProperty)
            autoCov[-1] = autoCov[-1]/(len(sequence)-i)

#Auto Covariance for ASA
#1 Feature
for ASAvals in ASAList:
    for i in range(lag):
        autoCov_ASA.append(0)
        feat_ASA.append('autoCovASA' + str(i))
        meanProperty = 0.0
        for AApos in range(len(ASAvals)):
            meanProperty += float(ASAvals[AApos])
        meanProperty = meanProperty/len(ASAvals)
        for AApos in range(len(ASAvals)-i):
            autoCov_ASA[-1] += (float(ASAvals[AApos]) - meanProperty) * (float(ASAvals[AApos + i]) - meanProperty)
        autoCov_ASA[-1] = autoCov_ASA[-1]/(len(ASAvals)-i)

# +++ Composition, Transition and Distribution +++
#CTD for sequence
numGroups = 7
for sequence in sequenceList:
    for AA in groupDict.keys():
        sequence = sequence.replace(AA, groupDict[AA])
    seqLen = len(sequence)
    regionList = [sequence[:int(seqLen/2)], sequence[int(seqLen/2):], sequence[:int(seqLen*0.25)], sequence[int(seqLen*0.25):int(seqLen*0.5)], sequence[int(seqLen*0.5):int(seqLen*0.75)], sequence[int(seqLen*0.75):], sequence[int(seqLen*0.25):int(seqLen*0.75)]]
    for region in regionList:
        CTD.append([[],[],[]])
        feat_CTD.append([[], [], []])
        for ID in range(numGroups):
            #Composition
            #49 Features
            CTD[-1][0].append(region.count(str(ID))/len(region))
            feat_CTD[-1][0].append('comp' + str(ID))
            #Distribution
            #245 Features, 62-306
            found = 0
            start = -1
            if region.count(str(ID)) == 0:
                for i in range(5):
                    CTD[-1][2].append(-1)
                    feat_CTD[-1][2].append('dist' + str(ID))
            while found < region.count(str(ID)):
                start = region.find(str(ID), start+1)
                found += 1
                if found == 1:
                    CTD[-1][2].append(start/len(region))
                    feat_CTD[-1][2].append('dist' + str(ID))
                if found == math.ceil(0.25*region.count(str(ID))):
                    CTD[-1][2].append(start/len(region))
                    feat_CTD[-1][2].append('dist' + str(ID))
                if found == math.ceil(0.5*region.count(str(ID))):
                    CTD[-1][2].append(start/len(region))
                    feat_CTD[-1][2].append('dist' + str(ID))
                if found == math.ceil(0.75*region.count(str(ID))):
                    CTD[-1][2].append(start/len(region))
                    feat_CTD[-1][2].append('dist' + str(ID))
                if found == region.count(str(ID)):
                    CTD[-1][2].append(start/len(region))
                    feat_CTD[-1][2].append('dist' + str(ID))
        #Transition
        #343 Features, 307-649
        for i in range(numGroups):
            for j in range(numGroups):
                CTD[-1][1].append(0)
                feat_CTD[-1][1].append('trans' + str(i) + '_' + str(j))
        for AA in range(len(region)-1):
            CTD[-1][1][int(region[AA])*numGroups+int(region[AA+1])]+=1/region.count(region[AA])

#CTD for SS
numGroups = 3
for secstruc in SSList:
    for SS in SSDict.keys():
        secstruc = secstruc.replace(SS, SSDict[SS])
    seqLen = len(secstruc)
    regionList = [secstruc[:int(seqLen/2)], secstruc[int(seqLen/2):], secstruc[:int(seqLen*0.25)], secstruc[int(seqLen*0.25):int(seqLen*0.5)], secstruc[int(seqLen*0.5):int(seqLen*0.75)], secstruc[int(seqLen*0.75):], secstruc[int(seqLen*0.25):int(seqLen*0.75)]]
    for region in regionList:
        CTD_SS.append([[],[],[]])
        feat_CTD_SS.append([[], [], []])
        for ID in range(numGroups):
            #Composition
            #21 Features
            CTD_SS[-1][0].append(region.count(str(ID))/len(region))
            feat_CTD_SS[-1][0].append('compSS' + str(ID))
            #Distribution
            #103 Features, but why??? why not 105??
            found = 0
            start = -1
            if region.count(str(ID)) == 0:
                for i in range(5):
                    CTD_SS[-1][2].append(-1)
                    feat_CTD_SS[-1][2].append('distSS' + str(ID))
            while found < region.count(str(ID)):
                start = region.find(str(ID), start+1)
                found += 1
                if found == 1:
                    CTD_SS[-1][2].append(start/len(region))
                    feat_CTD_SS[-1][2].append('distSS' + str(ID))
                if found == math.ceil(0.25*region.count(str(ID))):
                    CTD_SS[-1][2].append(start/len(region))
                    feat_CTD_SS[-1][2].append('distSS' + str(ID))
                if found == math.ceil(0.5*region.count(str(ID))):
                    CTD_SS[-1][2].append(start/len(region))
                    feat_CTD_SS[-1][2].append('distSS' + str(ID))
                if found == math.ceil(0.75*region.count(str(ID))):
                    CTD_SS[-1][2].append(start/len(region))
                    feat_CTD_SS[-1][2].append('distSS' + str(ID))
                if found == region.count(str(ID)):
                    CTD_SS[-1][2].append(start/len(region))
                    feat_CTD_SS[-1][2].append('distSS' + str(ID))
        #Transition
        #63 Features, 776-838
        for i in range(numGroups):
            for j in range(numGroups):
                CTD_SS[-1][1].append(0)
                feat_CTD_SS[-1][1].append('transSS' + str(i) + '_' + str(j))
        for SS in range(len(region)-1):
            CTD_SS[-1][1][int(region[SS])*numGroups+int(region[SS+1])]+=1/region.count(region[SS])

for i in range(len(IDList)):
    feature_index = 0
    outFile.write('#' + taxList[i].rstrip('\n') + ' ' + lenList[i] + '\n')
    outFile.write(str(cys_list[i]) + ' ')
    for j in range(len(autoCov)//len(IDList)):
        outFile.write(feat_AA[j+i*(len(autoCov)//len(IDList))] + '(' + str(feature_index) + '):' + str(autoCov[j+i*(len(autoCov)//len(IDList))]) + ' ')
        feature_index += 1
    for j in range(len(autoCov_ASA)//len(IDList)):
        outFile.write(feat_ASA[j+i*(len(autoCov_ASA)//len(IDList))] + '(' + str(feature_index) + '):' + str(autoCov_ASA[j+i*(len(autoCov_ASA)//len(IDList))]) + ' ')
        feature_index += 1
    for j in range(len(CTD)//len(IDList)):
        for k in range(len(CTD[j+i*(len(CTD)//len(IDList))])):
            for l in range(len(CTD[j+i*(len(CTD)//len(IDList))][k])):
                outFile.write(feat_CTD[j+i*(len(CTD)//len(IDList))][k][l] + '(' + str(feature_index) + '):' + str(CTD[j+i*(len(CTD)//len(IDList))][k][l]) + ' ')
                feature_index += 1
    for j in range(len(CTD_SS)//len(IDList)):
        for k in range(len(CTD_SS[j+i*(len(CTD_SS)//len(IDList))])):
            for l in range(len(CTD_SS[j+i*(len(CTD_SS)//len(IDList))][k])):
                outFile.write(feat_CTD_SS[j+i*(len(CTD_SS)//len(IDList))][k][l] + '(' + str(feature_index) + '):' + str(CTD_SS[j+i*(len(CTD)//len(IDList))][k][l]) + ' ')
                feature_index += 1
    outFile.write('\n')
