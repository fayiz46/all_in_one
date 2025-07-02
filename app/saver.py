# app/saver.py

import json

def save_to_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)
