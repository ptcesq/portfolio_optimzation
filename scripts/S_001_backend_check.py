# Quick script to check backend 

# libraries 
from qiskit_ibm_runtime import (
    QiskitRuntimeService
)

# Get credentials 
service = QiskitRuntimeService(
    channel="ibm_quantum_platform",
    instance="ptc_work", 
)

# Get Backends 
# Filter for real devices (exclude simulators) and those that are operational
backends = service.backends(
    simulator=False,
    operational=True
)

print("\n Backend \t Qubits \t Pending") 

# Print all 
for backend in backends: 
    print(f"{backend.name} \t {backend.num_qubits} \t\t {backend.status().pending_jobs}") 

# Sort by queue length
least_busy = sorted(backends, key=lambda b: b.status().pending_jobs)[0]

backend = least_busy
print('\n')
print(f"Least busy backend: {backend.name}")
print(f"Qubits: {backend.num_qubits}") 
print(f"Pending jobs: {backend.status().pending_jobs}")

