in_file = './ent_links'

import csv 

with open(in_file, 'r') as f:
    rows = [*csv.reader(f, delimiter='\t')]
rows = sorted(rows, key=lambda x: (x[0], x[1]))

with open(in_file + '_sorted.csv', 'w') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerows(rows)