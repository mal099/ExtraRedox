# Input:  xlsx file containing gene identifiers
# Output: redoxDB file
# Output may need some manual corrections

import pandas as pd
import urllib.parse
import urllib.request
import requests
import os

url = 'https://www.uniprot.org/uploadlists/'
genenames = []
rdbFile = open('../data/redoxDB_flavaPT.txt', 'w') # Output
NFFile = open('../data/redoxDB_NF.txt', 'w') # Debug File that shows which proteins were not found
descBool = False
taxBool = False
funcBool = False
pdbBool = False
seqBool = False
EC = 'NA\n'

data = pd.read_excel (r'../data/flava/aar2131_Table_S1.xlsx', sheet_name='3.PT') # Input
df = pd.DataFrame(data, columns= ['genes'])
for i in df['genes']:
    genenames.append(i.upper())

print(genenames)
for name in genenames:
    print('Finding Uniprot File ' + name + '...')
    try:
        if os.stat("../data/uniprot_flava/" + name + ".txt").st_size == 0 or os.stat("../data/uniprot_flava/" + name + ".fasta").st_size == 0:
            print(name + ' does not exist')
            NFFile.write(name + '\n')
            continue
        else:
            data = open("../data/uniprot_flava/" + name + ".txt")
    except:
        url = 'https://www.uniprot.org/uniprot/?query=reviewed:yes%20gene:' + name + '%20organism:human&format=txt&limit=1&sort=score'
        r = requests.get(url, allow_redirects=True)
        open("../data/uniprot_flava/" + name + ".txt", 'wb').write(r.content)
        url = 'https://www.uniprot.org/uniprot/?query=reviewed:yes%20gene:' + name + '%20organism:human&format=fasta&limit=1&sort=score'
        r = requests.get(url, allow_redirects=True)
        open("../data/uniprot_flava/" + name + ".fasta", 'wb').write(r.content)
        if os.stat("../data/uniprot_flava/" + name + ".txt").st_size == 0 or os.stat("../data/uniprot_flava/" + name + ".fasta").st_size == 0:
            url = 'https://www.uniprot.org/uniprot/?query=reviewed:yes%20gene:' + name + '&format=txt&limit=1&sort=score'
            r = requests.get(url, allow_redirects=True)
            open("../data/uniprot_flava/" + name + ".txt", 'wb').write(r.content)
            url = 'https://www.uniprot.org/uniprot/?query=reviewed:yes%20gene:' + name + '&format=fasta&limit=1&sort=score'
            r = requests.get(url, allow_redirects=True)
            open("../data/uniprot_flava/" + name + ".fasta", 'wb').write(r.content)
            NFFile.write(name + ' species\n')
            if os.stat("../data/uniprot_flava/" + name + ".txt").st_size == 0 or os.stat("../data/uniprot_flava/" + name + ".fasta").st_size == 0:
                url = 'https://www.uniprot.org/uniprot/?query=gene:' + name + '&format=txt&limit=1&sort=score'
                r = requests.get(url, allow_redirects=True)
                open("../data/uniprot_flava/" + name + ".txt", 'wb').write(r.content)
                url = 'https://www.uniprot.org/uniprot/?query=gene:' + name + '&format=fasta&limit=1&sort=score'
                r = requests.get(url, allow_redirects=True)
                open("../data/uniprot_flava/" + name + ".fasta", 'wb').write(r.content)
                NFFile.write(name + ' unreviewed\n')
                if os.stat("../data/uniprot_flava/" + name + ".txt").st_size == 0 or os.stat("../data/uniprot_flava/" + name + ".fasta").st_size == 0:
                    print(name + ' does not exist')
                    NFFile.write(name + '\n')
                    continue
                else:
                    data = open("../data/uniprot_flava/" + name + ".txt")
            else:
                data = open("../data/uniprot_flava/" + name + ".txt")
        else:
            data = open("../data/uniprot_flava/" + name + ".txt")
    for uni_line in data:
        if uni_line[0:2] == 'ID':
            rdbFile.write('ID: ' + name + ' ' + uni_line.split(' ')[-2] + ' aa\nGENENAME: ')
        elif uni_line[0:2] == 'GN' and '=' in uni_line:
            rdbFile.write(uni_line.rstrip('\n').split(';')[0].split('=')[1].split(' ')[0])
            if 'Synonyms=' in uni_line:
                uni_line = uni_line.rstrip('\n').split('Synonyms=')[1].split(',')
                for GN in uni_line:
                    rdbFile.write('; ' + GN.rstrip(';').strip(' '))
    rdbFile.write('\nDESCRIPTION: ')
    data = open("../data/uniprot_flava/" + name + ".txt")
    for uni_line in data:
        if uni_line[0:2] == 'DE' and '=' in uni_line and uni_line[14:16] != 'EC':
            descBool = True
            while uni_line.find('{') != -1:
                uni_line = uni_line[0:uni_line.find('{')] + uni_line[uni_line.find('}')+1:]
            rdbFile.write(uni_line.rstrip('\n').split('=')[1])
        elif uni_line[0:2] != 'DE' and descBool == True:
            descBool = False
            rdbFile.write('\n')
        elif uni_line[0:2] == 'OS':
            rdbFile.write('ORGANISM: ' + uni_line.split('   ')[1].rstrip('\n') + '\n')
        elif uni_line[0:2] == 'OC' and taxBool == False:
            taxBool = True
            rdbFile.write('TAXONOMY: ' + uni_line.split('   ')[1].rstrip('\n'))
        elif uni_line[0:2] == 'OC' and taxBool == True:
            rdbFile.write(uni_line.split('   ')[1].rstrip('\n'))
        elif uni_line[0:2] != 'OC' and taxBool == True:
            taxBool = False
            rdbFile.write('\n')
        elif uni_line[0:18] == 'CC   -!- FUNCTION:':
            funcBool = True
            while uni_line.find('{') != -1:
                uni_line = uni_line[0:uni_line.find('{')] + uni_line[uni_line.find('}')+1:]
            rdbFile.write(uni_line[9:].rstrip('\n'))
        elif funcBool == True and (uni_line[0:8] == 'CC   -!-' or uni_line[0:6] == 'CC   -'):
            funcBool = False
            rdbFile.write('\n')
            break
        elif uni_line[0:2] == 'CC' and funcBool == True and uni_line[9:12] != 'ECO':
            while uni_line.find('{') != -1 and uni_line.find('}') != -1:
                uni_line = uni_line[0:uni_line.find('{')] + uni_line[uni_line.find('}'):]
            while uni_line.find('{') != -1:
                uni_line = uni_line[0:uni_line.find('{')]
            while uni_line.find('}') != -1:
                uni_line = uni_line[uni_line.find('}')+1:]
            rdbFile.write(uni_line[8:].rstrip('\n'))
    rdbFile.write('UNIPROT: ')
    data = open("../data/uniprot_flava/" + name + ".txt")
    for uni_line in data:
        if uni_line[0:2] == 'AC':
            rdbFile.write(uni_line[5:].rstrip('\n'))
        elif uni_line[0:10] == 'DR   PDB; ' and pdbBool == True:
            while uni_line.find('{') != -1:
                uni_line = uni_line[0:uni_line.find('{')] + uni_line[uni_line.find('}')+1:]
            rdbFile.write(' & ' + uni_line[10:-2].replace(';', ','))
        elif uni_line[0:10] == 'DR   PDB; ':
            pdbBool = True
            while uni_line.find('{') != -1:
                uni_line = uni_line[0:uni_line.find('{')] + uni_line[uni_line.find('}')+1:]
            rdbFile.write('\nPDB: ' + uni_line[10:-2].replace(';', ','))
        elif uni_line[0:17] == 'DE            EC=':
            while uni_line.find('{') != -1:
                uni_line = uni_line[0:uni_line.find('{')] + uni_line[uni_line.find('}')+1:]
            EC = uni_line[17:]
            break
    if pdbBool == False:
        rdbFile.write('\nPDB: NA')
    rdbFile.write('\nEC: ' + EC)
    EC = 'NA\n'
    data = open("../data/uniprot_flava/" + name + ".txt")
    for uni_line in data:
        if 'Redox-active' in uni_line:
            rdbFile.write('CYSTEINE: ' + old_line)
            rdbFile.write('CYSTEINE: ' + uni_line)
        old_line = uni_line
    data = open("../data/uniprot_flava/" + name + ".txt")
    rdbFile.write('>' + name + '_HUMAN\n')
    for uni_line in data:
        if uni_line[0:2] == 'SQ':
            seqBool = True
            pdbBool = False
        elif seqBool == True and uni_line[0:2] == '  ':
            rdbFile.write(uni_line[5:].rstrip('\n').replace(' ', ''))
        elif uni_line[0:2] == '//':
            seqBool = False
            rdbFile.write('\n\n')
