
import sqlite3
conn = sqlite3.connect('data/answer_key/MIDI-B-Answer-Key-Validation.db')
rows = conn.execute('SELECT * FROM answer_data').fetchall()
col1_vals = set(r[1] for r in rows)
col2_vals = set(r[2] for r in rows)
col4_vals = set(r[4] for r in rows)
print('col[1] unique values:', col1_vals)
print('col[2] modalities:', col2_vals)
print('col[4] unique patient IDs:', len(col4_vals))
print('Sample patient IDs:', list(col4_vals)[:5])
