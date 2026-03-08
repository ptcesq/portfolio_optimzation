from importlib.metadata import version
from tabulate import tabulate
import sys

libs = ["numpy", 
    "torch", 
    "matplotlib", 
    "seaborn", 
    "qiskit", 
    "qiskit_machine_learning", 
    "qiskit_optimization", 
    "qiskit_ibm_runtime", 
    "pylatexenc", 
    "qiskit_algorithms", 
    "yfinance", 
    "qiskit_aer"] 

def environment_state():
    data = []
    headers = ['System/Library', 'Version'] 
    data.append(['Python', sys.version.split()[0]])
    for package in libs:
        try: 
            ver = version(package) 
            data.append([package, ver]) 
        except:
            ver = "Not Imported"
    return data, headers 