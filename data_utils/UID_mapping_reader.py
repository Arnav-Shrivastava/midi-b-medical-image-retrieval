
import csv
with open('data/answer_key/MIDI-B-UID-Mapping-Validation.csv') as f:
    rows = list(csv.reader(f))
print('Columns:', rows[0])
print()
for r in rows[1:4]:
    print(r)
print('Total rows:', len(rows)-1)
