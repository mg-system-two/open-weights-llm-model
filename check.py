import logging
logging.disable(logging.CRITICAL)
from arcadedb_client import MaterialsDB
db = MaterialsDB()
db._ensure_database()

print('=== MATERIALS (sample) ===')
r = db.sql('SELECT name FROM Material LIMIT 20')
for x in r: print(' ', x.get('name'))

print()
print('=== PROCESSES (sample) ===')
r = db.sql('SELECT name FROM Process LIMIT 20')
for x in r: print(' ', x.get('name'))

print()
print('=== APPLICATIONS (sample) ===')
r = db.sql('SELECT name FROM Application LIMIT 20')
for x in r: print(' ', x.get('name'))
