# Input: identifier file from uniprot
# Output: redoxDB file, finding all proteins in the file with a redox modification
# Output may need some manual corrections

import pandas as pd
import os
import re

rdbFile = open('../data/redoxDB_uniprot.txt', 'w') # Output
note_list = ['Redox-active', 'Cysteine sulfenic acid (-SOH)', 'Cysteine sulfinic acid (-SO2H)', 'Cysteine sulfonic acid (-SO3H)', 'S-nitrosocysteine', 'S-glutathionyl cysteine']

from Bio import SwissProt
for record in SwissProt.parse(open('../data/uniprot/uniprot_sprot.dat')): # Input
    if 'Eukaryota' in record.organism_classification:
        for feature in record.features:
            if (feature.type == 'DISULFID' or feature.type == 'MOD_RES') and 'note' in feature.qualifiers.keys() and feature.qualifiers['note'] in note_list:
                print('Writing ' + record.entry_name + '...')
                rdbFile.write('ID: ' + record.entry_name + ' ' + str(record.sequence_length) + ' aa\n')
                rdbFile.write('GENENAME: ' + record.gene_name[record.gene_name.find('Name=')+5:].split(' ')[0][:-1])
                gene_list = record.gene_name[record.gene_name.find('Synonyms=')+9:].split(';')[0].split(', ')
                for gene_name in gene_list:
                    rdbFile.write('; ' + gene_name.split('{')[0])
                rdbFile.write('\n')
                rdbFile.write('DESCRIPTION: ')
                description_string = re.sub("[\(\{].*?[\)\}]", "", record.description)
                description_rec = description_string.split('Flags: ')[0].split('EC=')[0].split(' AltName')[0]
                rec_string = ''
                for description_part in description_rec.rstrip().rstrip(';').split('; '):
                    rec_string += description_part.split('=')[1].split('{')[0].rstrip() + '; '
                rdbFile.write(rec_string[:-2])
                description_list = description_string[:description_string.find('Flags: ')].split(' AltName: ')[1:]
                for description in description_list:
                    for description_part in description.rstrip().rstrip(';').split('; '):
                        rdbFile.write('; ' + description_part.split('=')[1].split('{')[0].rstrip())
                rdbFile.write('\n')
                rdbFile.write('ORGANISM: ' + record.organism.rstrip('.') + '\n')
                rdbFile.write('TAXONOMY: ' + record.organism_classification[0])
                for organism in record.organism_classification[1:]:
                    rdbFile.write('; ' + organism)
                rdbFile.write('\n')
                function_string = re.sub("[\(\{].*?[\)\}]", "", record.comments[0])
                rdbFile.write(function_string.rstrip('. ') + '.\n')
                rdbFile.write('UNIPROT: ')
                uniprot_string = ''
                for access in record.accessions:
                    uniprot_string += access + '; '
                rdbFile.write(uniprot_string[:-2] + '\n')
                rdbFile.write('PDB: ')
                pdb_string = ''
                pdb_bool = False
                for PDB_ID in record.cross_references:
                    if PDB_ID[0] == 'PDB':
                        pdb_bool = True
                        for pdb_data in PDB_ID[1:]:
                            pdb_string += pdb_data + ', '
                        pdb_string = pdb_string.rstrip(', ')
                        pdb_string += ' & '
                if not pdb_bool: pdb_string = 'NA'
                rdbFile.write(pdb_string.rstrip(' &') + '\n')
                if record.description.find('EC=') == -1:
                    rdbFile.write('EC: NA\n')
                else:
                    rdbFile.write('EC: ' + record.description[record.description.find('EC=')+3:].split(' ')[0] + '\n')
                cysteine_list = []
                modification_string = ''
                for feature in record.features:#!change mod type to the types in redoxdb
                    if (feature.type == 'DISULFID' or feature.type == 'MOD_RES') and 'note' in feature.qualifiers.keys() and feature.qualifiers['note'] in note_list:
                        if feature.type == 'DISULFID':
                            cysteine_list = cysteine_list + [[feature.location.start, 'disulfide'], [feature.location.end-1, 'disulfide']]
                            modification_string += 'disulfide&'
                        else:
                            cysteine_list.append([feature.location.start, feature.qualifiers['note']])
                            modification_string += feature.qualifiers['note'] + '&'
                for cysteine in cysteine_list:
                    rdbFile.write('CYSTEINE: ' + str(cysteine[0]) + ', ' + cysteine[1] + ', NA, NA\n')
                rdbFile.write('SOURCE: UNIPROT\n')
                rdbFile.write('MODIFICATION: ' + modification_string.rstrip('&') + '\n')
                rdbFile.write('REFERENCE: NA\n')
                rdbFile.write('>' + record.entry_name + '\n')
                rdbFile.write(record.sequence + '\n\n')
                break
