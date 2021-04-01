# This file can be used to extract statistics from a large number of proteins
# regarding their AAs, SSs, HSE, accessibility. Needs input from predict_[x] file.

import numpy as np
from scipy import stats
import math
import matplotlib.pyplot as plt
from matplotlib.pyplot import Axes
import pandas as pd

# Input/Output
inFile = open('../data/dev_output_thesis_seq.txt')
#inFile = open('../data/mammal_dist.txt')
#inFile = open('../data/cI_paper.txt')
modFile = open("../data/paper_stats_mod_all.txt", "w")
unmodFile = open("../data/paper_stats_unmod_all.txt", "w")
jorkFile = open("../data/jork.txt", "w")
u_file = open("../data/u_file.txt", "w")
seqFile = open('../data/cys_logos_all.txt')
noseqFile = open('../data/nomod_logos_all.txt')

# Setup lists
cys_mod = [[], [], [], [], [], [], [], [], [], []] # SS, concav, pKa, pka3, HSE1, HSE2, DSSP accessibility, phospho, acetyl, ubiquitin
cys_unmod = [[], [], [], [], [], [], [], [], [], []]
aa_mod = [[], [], [], [], []] # AA, SS, HSE1, HSE2, DSSP accessibility
aa_unmod = [[], [], [], [], []]

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

AA_ratio_dict = {'A': 0.9387869871508409, 'C': 0.9980338183248132, 'D': 0.9507620987567605, 'E': 1.0728587164795291, 'F': 0.8500182681768359, 'G': 1.243530552119071, 'H': 1.1984308597580908, 'I': 0.8935046646717127, 'K': 1.232671741465465, 'L': 0.6976523844615222, 'M': 0.8176891874762328, 'N': 0.9839188271682552, 'P': 1.3844150912690878, 'Q': 0.8433402346445824, 'R': 1.0717263677358972, 'S': 1.2478970490914454, 'T': 1.2112311051822144, 'V': 0.9376268718704939, 'W': 1.10624387054593, 'Y': 0.8943188945313288, 'X': 1.0, 'U': 0.9980338183248132, '-': 1.0}
SS_ratio_dict = {'e': 0.9555965191916798, 'h': 0.8554949733268717, ' ': 1.1446746637586331, 's': 1.3007758115551387, 't': 1.0889975368658358, 'b': 0.909460105112279, 'g': 1.0635681300489177, 'i': 1.6469824789097987}

def show_stats(modlist, unmodlist, name):
    print('\nStats for ' + name + ':')
    print('Mean Cys+: ' + str(np.mean(modlist)))
    print('Standard Dev Cys+: ' + str(np.std(modlist)))
    print('Median Cys+: ' + str(np.median(modlist)))
    print('IQR Cys+: ' + str(np.quantile(modlist, .75)) + ' - ' + str(np.quantile(modlist, .25)) + ' = ' + str(np.quantile(modlist, .75)-np.quantile(modlist, .25)))
    print('Mean Cys-: ' + str(np.mean(unmodlist)))
    print('Standard Dev Cys-: ' + str(np.std(unmodlist)))
    print('Median Cys-: ' + str(np.median(unmodlist)))
    print('IQR Cys-: ' + str(np.quantile(unmodlist, .75)) + ' - ' + str(np.quantile(unmodlist, .25)) + ' = ' + str(np.quantile(unmodlist, .75)-np.quantile(unmodlist, .25)))
    print('t-test: ' + str(stats.mannwhitneyu(modlist, unmodlist)))
    print('Ratio Means: ' + str(np.mean(modlist)/np.mean(unmodlist)))
    print('Ratio Medians: ' + str(np.median(modlist)/np.median(unmodlist)))
    print('Poisson: ' + str(np.sqrt(np.sum(modlist[0:len(unmodlist)]))/np.sum(unmodlist[0:len(modlist)])))


def jorg_plot(mod_data, unmod_data, name, range_number = 40, range_max = 1.0):
    range_width = range_max/range_number
    range_mod = []
    range_list = []
    range_ratio = []
    for i in range(range_number):
        range_list.append(i*range_width)
        range_mod.append([])
        for j in mod_data:
            if j >= i*range_width:
                range_mod[i].append(j)
    range_unmod = []
    for i in range(range_number):
        range_unmod.append([])
        for j in unmod_data:
            if j >= i*range_width:
                range_unmod[i].append(j)
    significance_int = 0
    for i in range(len(range_mod)):
        if stats.mannwhitneyu(range_mod[i], range_unmod[i])[1] > 0.05 and significance_int == 0:
            significance_int = i
        if len(range_unmod[i]) != 0:
            range_ratio.append(len(range_mod[i])/len(range_unmod[i]))
    plt.plot(range_list[:significance_int], range_ratio[:significance_int], color='green', linestyle='solid', linewidth=2, markersize=12)
    plt.plot(range_list[significance_int-1:len(range_ratio)], range_ratio[significance_int-1:], color='green', linestyle='dashed', linewidth=2, markersize=12)
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    plt.ylabel('P', fontsize=14, fontweight='bold', rotation = 90)
    plt.xlabel(name, fontsize=14, fontweight='bold')
    #plt.ylim(top=1, bottom=0)
    plt.show()
    plt.clf()

def reject_outliers(sr, iq_range=0.5):
    pcnt = (1 - iq_range) / 2
    qlow, median, qhigh = sr.dropna().quantile([pcnt, 0.50, 1-pcnt])
    iqr = qhigh - qlow
    return sr[ (sr - median).abs() <= iqr]

def draw_boxplot(data_list, label_list, ystring):
    plt.boxplot(data_list, labels=label_list, showmeans=True, notch=True)
    plt.xticks(fontsize=24, fontweight='bold')
    plt.yticks(fontsize=24, fontweight='bold')
    plt.ylabel(ystring, fontsize=28, fontweight='bold')
    plt.show()
    plt.clf()

def draw_barplot(xlist, modlist, unmodlist, name, labels):
    p_list_neg = []
    p_list_pos = []
    xlist_star = [[], []]
    p_values = []
    error = []
    ratio_means = []
    for i in range(len(xlist)):
        stats_mod = []
        stats_unmod = []
        for mod_element in modlist:
            stats_mod.append(mod_element.count(xlist[i]))
        for unmod_element in unmodlist:
            stats_unmod.append(unmod_element.count(xlist[i]))
        show_stats(stats_mod, stats_unmod, name + ' ' + str(xlist[i]))
        error.append(3*np.sqrt(np.sum(stats_mod[0:len(stats_unmod)]))/np.sum(stats_unmod[0:len(stats_mod)]))
        ratio_means.append(np.mean(stats_mod)/np.mean(stats_unmod))
        p_values.append([stats.mannwhitneyu(stats_mod, stats_unmod)[1], i])
    p_values.sort()
    for i in range(len(p_values)):
        p_values[i][0] = p_values[i][0] * (20-i)
    p_values.sort(key=lambda x: x[1])
    for i in range(len(xlist)):
        if p_values[i][0] < 0.05:
            if ratio_means[i] < 1.0:
                xlist_star[1].append('#e84e4e')
                if p_values[i][0] < 0.001:
                    xlist_star[0].append('***')
                elif p_values[i][0] < 0.01:
                    xlist_star[0].append('**')
                else:
                    xlist_star[0].append('*')
            else:
                xlist_star[1].append('#7acb6d')
                if p_values[i][0] < 0.001:
                    xlist_star[0].append('***')
                elif p_values[i][0] < 0.01:
                    xlist_star[0].append('**')
                else:
                    xlist_star[0].append('*')
        else:
            xlist_star[1].append('#878787') ##377bf9 blue
            xlist_star[0].append('')
    x = np.arange(len(xlist))
    bar1 = plt.bar(x, ratio_means, color = xlist_star[1], yerr = error, capsize=3)
    for i in range(len(bar1)):
        height = bar1[i].get_height() + error[i]
        plt.text(bar1[i].get_x() + bar1[i].get_width()/2.0, height, xlist_star[0][i], ha='center', va='bottom', fontsize=14, fontweight='bold')
    plt.xticks(x, xlist, fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    plt.ylabel('Ratio Cys+/Cys-', fontsize=14, fontweight='bold')
    plt.xlabel(name, fontsize=14, fontweight='bold')
    plt.show()
    plt.clf()

# Go through data
for line in inFile:
    line = line.rstrip().split('$')[1]
    if len(line)>100: #check if data is complete
        # Split data into modified and unmodified
        if line[0] == '0':
            # Handle cysteine data
            cys_unmod[0].append(line[2])
            cys_unmod[1].append(float(line[4:].split(' ')[0]))
            if line[4:].split(' ')[1] != 'N/A':
                cys_unmod[2].append(float(line[4:].split(' ')[1])) #pka
            if line[4:].split(' ')[2] != 'N/A':
                cys_unmod[3].append(float(line[4:].split(' ')[2])) #pka3
            exposure = line.split('(')[1].split(')')[0].split(',')
            cys_unmod[4].append(float(exposure[0]))
            cys_unmod[5].append(float(exposure[1][1:]))
            cys_unmod[7].append(int(line.split('!')[0][-5]))
            cys_unmod[8].append(int(line.split('!')[0][-3]))
            cys_unmod[9].append(int(line.split('!')[0][-1]))
            if float(line.split(')')[1].split(' ')[1].strip()) > 1.0:
                #dssp_access = 1.0
                dssp_access = float(line.split(')')[1].split(' ')[1].strip())
                #dssp_access = 0.0
            else:
                dssp_access = float(line.split(')')[1].split(' ')[1].strip())
                #dssp_access = math.log(float(line.split(')')[1].split('!')[0].strip())+0.001, 1.2)
            cys_unmod[6].append(dssp_access)
            #cys_unmod[4].append(math.log(float(line.split(')')[1].split('!')[0].strip())+0.001, 2))
            # Handle other data
            line = line.split('!')[2:-1]
            aa_unmod[0].append([])
            aa_unmod[1].append([])
            for j in range(19):#len(line)):
                AAvals = line[j].split('&')
                aa_unmod[0][-1].append(AAvals[1])
                aa_unmod[1][-1].append(AAvals[0])
                exposure = line[j].split('(')[1].split(')')[0].split(',')
                aa_unmod[2].append(float(exposure[0]))
                aa_unmod[3].append(float(exposure[1][1:]))
                if float(AAvals[3]) > 1.0:
                    dssp_access = 1.0
                else:
                    dssp_access = float(AAvals[3])
                aa_unmod[4].append(dssp_access)
        elif line[0] == '1':
            # Handle cysteine data
            cys_mod[0].append(line[2]) #SS
            cys_mod[1].append(float(line[4:].split(' ')[0])) #concav
            if line[4:].split(' ')[1] != 'N/A':
                cys_mod[2].append(float(line[4:].split(' ')[1])) #pka
            if line[4:].split(' ')[2] != 'N/A':
                cys_mod[3].append(float(line[4:].split(' ')[2])) #pka3
            exposure = line.split('(')[1].split(')')[0].split(',')
            cys_mod[4].append(float(exposure[0])) #HSE1
            cys_mod[5].append(float(exposure[1][1:])) #HSE2
            cys_mod[7].append(int(line.split('!')[0][-5])) #phospho
            cys_mod[8].append(int(line.split('!')[0][-3])) #acetyl
            cys_mod[9].append(int(line.split('!')[0][-1])) #ubiquitin
            if float(line.split(')')[1].split(' ')[1].strip()) > 1.0:
                #dssp_access = 1.0
                dssp_access = float(line.split(')')[1].split(' ')[1].strip())
                #dssp_access = 0.0
            else:
                dssp_access = float(line.split(')')[1].split(' ')[1].strip())
                #dssp_access = math.log(float(line.split(')')[1].split('!')[0].strip())+0.001, 1.2)
            cys_mod[6].append(dssp_access)
            # Handle other data
            line = line.split('!')[2:-1]
            aa_mod[0].append([])
            aa_mod[1].append([])
            for j in range(19):#len(line)):
                AAvals = line[j].split('&')
                aa_mod[0][-1].append(AAvals[1])
                aa_mod[1][-1].append(AAvals[0])
                exposure = line[j].split('(')[1].split(')')[0].split(',')
                aa_mod[2].append(float(exposure[0]))
                aa_mod[3].append(float(exposure[1][1:]))
                if float(AAvals[3]) > 1.0:
                    dssp_access = 1.0
                else:
                    dssp_access = float(AAvals[3])
                aa_mod[4].append(dssp_access)
        else:
            raise Exception('x should be 0 or 1. The value of x was: {}'.format(line[0]))

# Go through seq data
seqList = []
noseqList = []
for line in seqFile:
    line = line.rstrip()
    line = line[0:10] + line[11:]
    seqList.append(line)
for line in noseqFile:
    line = line.rstrip()
    line = line[0:10] + line[11:]
    noseqList.append(line)

AAlist = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
draw_barplot(AAlist, seqList, noseqList, 'Amino Acid (Seq)', AAlist)

#write data into file
modFile.write('Cysteines\n')
for i in cys_mod:
    for j in i:
        modFile.write(str(j) + '\n')
    modFile.write('\n')
modFile.write('Amino Acids\n')
for i in aa_mod[0]:
    dict_sum = 0.0
    for j in i:
        modFile.write(str(j))
        dict_sum += AA_ratio_dict[j]
    modFile.write('\n' + str(dict_sum))
    modFile.write('\n')
modFile.write('\n')
for i in aa_mod[1]:
    dict_sum = 0.0
    for j in i:
        modFile.write(str(j))
        dict_sum += SS_ratio_dict[j]
    modFile.write('\n' + str(dict_sum))
    modFile.write('\n')
modFile.write('\n')
for i in aa_mod[2:]:
    for j in i:
        modFile.write(str(j) + '\n')
    modFile.write('\n')

unmodFile.write('Cysteines\n')
for i in cys_unmod:
    for j in i:
        unmodFile.write(str(j) + '\n')
    unmodFile.write('\n')
unmodFile.write('Amino Acids\n')
for i in aa_unmod[0]:
    dict_sum = 0.0
    for j in i:
        unmodFile.write(str(j))
        dict_sum += AA_ratio_dict[j]
    unmodFile.write('\n' + str(dict_sum))
    unmodFile.write('\n')
unmodFile.write('\n')
for i in aa_unmod[1]:
    dict_sum = 0.0
    for j in i:
        unmodFile.write(str(j))
        dict_sum += SS_ratio_dict[j]
    unmodFile.write('\n' + str(dict_sum))
    unmodFile.write('\n')
unmodFile.write('\n')
for i in aa_unmod[2:]:
    for j in i:
        unmodFile.write(str(j) + '\n')
    unmodFile.write('\n')

# Amino Acid
AAlist = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
draw_barplot(AAlist, aa_mod[0], aa_unmod[0], 'Amino Acid', AAlist)

# Amino Acid features
features = ['Mass', 'Vol', 'Area', 'SEA1', 'SEA2', 'SEA3', 'Alpha', 'Beta', 'Turn', 'Polar', 'Non-P', 'Charge', 'Positive', 'Tiny', 'Small', 'Aromatic', 'Aliphatic']
AAlist_star = [[], []]
feature_list_mod = []
feature_list_unmod = []
Feature_ratio_means = []
p_values = []
numberFeat = len(AAdict['A'])
for feature in range(numberFeat):
    feature_list_mod.append([])
    feature_list_unmod.append([])
    for peptide in aa_mod[0]:
        feature_list_mod[-1].append(0)
        for aa in peptide:
            feature_list_mod[-1][-1] += AAdict[aa][feature]
    for peptide in aa_unmod[0]:
        feature_list_unmod[-1].append(0)
        for aa in peptide:
            feature_list_unmod[-1][-1] += AAdict[aa][feature]
    p_values.append([stats.mannwhitneyu(feature_list_mod[-1], feature_list_unmod[-1])[1], feature])
    u_file.write(features[feature] + '\t' + str(stats.mannwhitneyu(feature_list_mod[-1], feature_list_unmod[-1])[0]) + '\n')
u_file.write('n1 = ' + str(len(cys_mod[4])) + ' n2 = ' + str(len(cys_unmod[4])))
p_values.sort()
for i in range(len(p_values)):
    p_values[i][0] = p_values[i][0] * (17-i)
p_values.sort(key=lambda x: x[1])
jorkFile.write('Cys+\n')
for i in range(len(features)):
    jorkFile.write(features[i] + ': ')
    for j in range(len(feature_list_mod[0])):
        jorkFile.write(str(feature_list_mod[i][j]) + ', ')
    jorkFile.write('\n')
jorkFile.write('Cys-\n')
for i in range(len(features)):
    jorkFile.write(features[i] + ': ')
    for j in range(len(feature_list_unmod[0])):
        jorkFile.write(str(feature_list_unmod[i][j]) + ', ')
    jorkFile.write('\n')
error = []
for i in range(len(feature_list_mod)):
    show_stats(feature_list_mod[i], feature_list_unmod[i], 'Feature ' + str(features[i]))
    error.append(3*np.sqrt(np.sum(feature_list_mod[i][0:len(feature_list_unmod[i])]))/np.sum(feature_list_unmod[i][0:len(feature_list_mod[i])]))
    if p_values[i][0] < 0.05:
        if ((sum(feature_list_mod[i])/len(aa_mod[0]))/(sum(feature_list_unmod[i])/len(aa_unmod[0]))) < 1.0:
            AAlist_star[1].append('#e84e4e')
            if p_values[i][0] < 0.001:
                AAlist_star[0].append('***')
            elif p_values[i][0] < 0.01:
                AAlist_star[0].append('**')
            else:
                AAlist_star[0].append('*')
        else:
            AAlist_star[1].append('#7acb6d')
            if p_values[i][0] < 0.001:
                AAlist_star[0].append('***')
            elif p_values[i][0] < 0.01:
                AAlist_star[0].append('**')
            else:
                AAlist_star[0].append('*')
    else:
        AAlist_star[1].append('#878787')
        AAlist_star[0].append('')
    feature_list_mod[i].sort()
    new_list = feature_list_unmod[i][:len(feature_list_mod[i])]
    new_list.sort()
    Feature_ratio_means.append((sum(feature_list_mod[i])/len(aa_mod[0]))/(sum(feature_list_unmod[i])/len(aa_unmod[0])))
x = np.arange(len(features))
bar1 = plt.bar(x, Feature_ratio_means, color = AAlist_star[1], yerr = error, capsize=3)
for i in range(len(bar1)):
    height = bar1[i].get_height() + error[i]
    plt.text(bar1[i].get_x() + bar1[i].get_width()/2.0, height, AAlist_star[0][i], ha='center', va='bottom', fontsize=14, fontweight='bold')
plt.xticks(x, features, rotation = 30, fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')
plt.ylabel('Ratio Cys+/Cys-', fontsize=14, fontweight='bold')
plt.xlabel('Feature', fontsize=14, fontweight='bold')
plt.show()
plt.clf()

# Secondary Structure
SSlist = ('e','h',' ','s','t','b','g','i')
SSlabels = ('\u03B2-strand','\u03B1-helix','loop','bend','turn','\u03B2-bridge','3-helix','5-helix')
draw_barplot(SSlist, aa_mod[1], aa_unmod[1], 'Secondary structure', SSlabels)

# draw boxplots for everything
draw_boxplot([cys_mod[1], cys_unmod[1]], ['Cys+', 'Cys-'], 'Cysteine ConCavity')
draw_boxplot([cys_mod[2], cys_unmod[2]], ['Cys+', 'Cys-'], 'Cysteine pKa')
draw_boxplot([cys_mod[3], cys_unmod[3]], ['Cys+', 'Cys-'], 'Cysteine pKa')
draw_boxplot([cys_mod[6], cys_unmod[6], aa_mod[4], aa_unmod[4]], ['Cys+', 'Cys-', 'AA+', 'AA-'], 'Cysteine DSSP accessibility')
draw_boxplot([cys_mod[4], cys_unmod[4], aa_mod[2], aa_unmod[2]], ['Cys+', 'Cys-', 'AA+', 'AA-'], 'Cysteine HSE1')
draw_boxplot([cys_mod[5], cys_unmod[5], aa_mod[3], aa_unmod[3]], ['Cys+', 'Cys-', 'AA+', 'AA-'], 'Cysteine HSE2')
draw_boxplot([cys_mod[7], cys_unmod[7]], ['Cys+', 'Cys-'], 'Cysteine phospho')
draw_boxplot([cys_mod[8], cys_unmod[8]], ['Cys+', 'Cys-'], 'Cysteine acetyl')
draw_boxplot([cys_mod[9], cys_unmod[9]], ['Cys+', 'Cys-'], 'Cysteine ubiquitin')

# show stats for everything
show_stats(cys_mod[1], cys_unmod[1], 'Cysteine ConCavity')
show_stats(cys_mod[2], cys_unmod[2], 'Cysteine pKa2')
show_stats(cys_mod[3], cys_unmod[3], 'Cysteine pKa3')
show_stats(cys_mod[6], cys_unmod[6], 'Cysteine DSSP accessibility')
show_stats(cys_mod[4], cys_unmod[4], 'Cysteine HSE1')
show_stats(cys_mod[5], cys_unmod[5], 'Cysteine HSE2')
show_stats(cys_mod[7], cys_unmod[7], 'Cysteine phospho')
show_stats(cys_mod[8], cys_unmod[8], 'Cysteine acetyl')
show_stats(cys_mod[9], cys_unmod[9], 'Cysteine ubiquitin')
show_stats(aa_mod[4], aa_unmod[4], 'AA DSSP accessibility')
show_stats(aa_mod[2], aa_unmod[2], 'AA HSE1')
show_stats(aa_mod[3], aa_unmod[3], 'AA HSE2')

# remove outliers
for i in range(9):
    cys_mod[i+1] = reject_outliers(pd.Series(cys_mod[i+1]))
    cys_unmod[i+1] = reject_outliers(pd.Series(cys_unmod[i+1]))

# draw boxplots with no outliers
draw_boxplot([cys_mod[1], cys_unmod[1]], ['Cys+', 'Cys-'], 'Cysteine ConCavity')
draw_boxplot([cys_mod[2], cys_unmod[2]], ['Cys+', 'Cys-'], 'Cysteine pKa')
draw_boxplot([cys_mod[3], cys_unmod[3]], ['Cys+', 'Cys-'], 'Cysteine pKa')
draw_boxplot([cys_mod[6], cys_unmod[6], aa_mod[4], aa_unmod[4]], ['Cys+', 'Cys-', 'AA+', 'AA-'], 'Cysteine DSSP accessibility')
draw_boxplot([cys_mod[4], cys_unmod[4], aa_mod[2], aa_unmod[2]], ['Cys+', 'Cys-', 'AA+', 'AA-'], 'Cysteine HSE1')
draw_boxplot([cys_mod[5], cys_unmod[5], aa_mod[3], aa_unmod[3]], ['Cys+', 'Cys-', 'AA+', 'AA-'], 'Cysteine HSE2')
draw_boxplot([cys_mod[7], cys_unmod[7]], ['Cys+', 'Cys-'], 'Cysteine phospho')
draw_boxplot([cys_mod[8], cys_unmod[8]], ['Cys+', 'Cys-'], 'Cysteine acetyl')
draw_boxplot([cys_mod[9], cys_unmod[9]], ['Cys+', 'Cys-'], 'Cysteine ubiquitin')

#jorg_plot(cys_mod[6], cys_unmod[6], 'Cysteine accessibility')
'''fig, ax = plt.subplots(figsize=(8, 4))
n, bins, patches = ax.hist(cys_unmod[2], 50, density=True, histtype='step',
                           cumulative=True, label='Unmodified')
ax.hist(cys_mod[4], 50, density=True, histtype='step',
                           cumulative=True, label='Modified')
ax.set_xlim(0, 29)
plt.ylabel('Relative Cumulative Frequency')
plt.xlabel('HSE1 of cysteines')
plt.show()
plt.clf()'''
