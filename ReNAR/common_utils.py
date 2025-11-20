import json, os, datetime, copy
from typing import Any, Dict, List, Tuple
from pathlib import Path
base_path = ''

def read_json(path:str):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    else:
        raise FileNotFoundError(f"File {path} does not exist.")

def write_json(path:str, data):
    file_path = Path(path)
    
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def get_time():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
def record_results(result_path:str, algo, stats, label:str, setting:Dict[str,str]):
    if os.path.exists(result_path):
        data = read_json(result_path)
    else:
        data = {'config':setting,
                'best':{},
                'results':{}}
        write_json(result_path, data)
    results = data['results'].copy()
    if algo not in results:
        results[algo] = {}
    if label not in results[algo]:
        results[algo][label] = {}
    results[algo][label] = stats['score']
    data['results'] = results
    data['best'][algo] = max(data['results'][algo].values())
    data['config']['algo_counts'] = len(data['best'].keys())
    if len(data['config']['algorithms']) < data['config']['algo_counts']:
        data['config']['algorithms'] = list(data['results'].keys())
    write_json(result_path, data)
        