from qiskit import transpile, QuantumCircuit, QuantumRegister, Aer
from qiskit.circuit import Parameter
from qiskit.ignis.verification.tomography import state_tomography_circuits
from isl.recompilers import ISLConfig, ISLRecompiler
import numpy as np

from file_utils import *

# load statevector backend
statevector = Aer.get_backend('statevector_simulator')

# Parameterize variable t to be evaluated at t=pi later
t = Parameter('t')



### Recompilation tools


def ISL(qc, filename, sufficient_cost=1e-2):
    '''Uniquely identify ISL recompiled Trotter circuit : num qubits, sufficient cost, trotter steps, final evol time
    
    Pass in filename? If exists, return that, else do the recompilation then save the resulting qasm string using
    the input filename.
    
    Args:
        qc (QuantumCircuit) : quantum circuit to be recompiled.
        sufficient_cost (float) : Termination condition for recompilation. Default 1e-2. 

    Returns:
        Dict : an ISL results object of the form:
             result_dict = {
                "circuit": QuantumCircuit,
                "overlap": float,
                "exact_overlap": float,
                "num_1q_gates": int,
                "num_2q_gates": int,
                "cost_progression": List[float],
                "entanglement_measures_progression": List[float],
                "e_val_history": List[float],
                "qubit_pair_history": List[Tuple[int, int]],
                "method_history": List[string],
                "time_taken": float,
                "cost_evaluations": int,
                "coupling_map": List[Tuple[int, int]],
                "circuit_qasm": string
            }
    '''
    config = ISLConfig(sufficient_cost=sufficient_cost)
        
    try: # try and load previously recompiled circuit
        result = np.load(filename + '.npy', allow_pickle=True).tolist()
        print('existing ISL recompiled circuit found!')
    except FileNotFoundError: # do recompilation from scratch and save result
        print('New circuit recompilation being done with ISL')
        recompiler = ISLRecompiler(transpile(qc, basis_gates=['u1','u2','u3','cx']), isl_config=config, backend=statevector)
        result = recompiler.recompile()
        np.save(filename + '.npy', result)
    
    return result

def ISL_from_scratch(qc, sufficient_cost=1e-2):
    '''A wrapper for the ISL recompiler which won't save the result to file.
    
    Args:
        qc (QuantumCircuit) : quantum circuit to be recompiled.
        sufficient_cost (float) : Termination condition for recompilation. Default 1e-2. 

    Returns:
        Dict : an ISL results object of the form:
             result_dict = {
                "circuit": QuantumCircuit,
                "overlap": float,
                "exact_overlap": float,
                "num_1q_gates": int,
                "num_2q_gates": int,
                "cost_progression": List[float],
                "entanglement_measures_progression": List[float],
                "e_val_history": List[float],
                "qubit_pair_history": List[Tuple[int, int]],
                "method_history": List[string],
                "time_taken": float,
                "cost_evaluations": int,
                "coupling_map": List[Tuple[int, int]],
                "circuit_qasm": string
            }
    '''
    config = ISLConfig(sufficient_cost=sufficient_cost)
    recompiler = ISLRecompiler(transpile(qc, basis_gates=['u1','u2','u3','cx']), isl_config=config, backend=statevector)
 
    return recompiler.recompile()



### Trotter tools
def build_trotter_circuit(qc, quantum_register, target_qubit_indices, trotterisation, trotter_steps, time_slice):
    '''Args:
            qc (QuantumCircuit) : quantum circuit to be recompiled.
            quantum_register (QuantumRegister) : quantum register
            target_qubit_indices (List[int]) : List of qubits to apply trotterisation to
            trotterisation (QuantumCircuit) : Chosen trotterisation of target Hamiltonian
            trotter_steps (int) : number of repeats of trotterisation to use.
            time_slice (float) : the time step covered by each trotter step. 

        Returns
            QuantumCircuit
    '''
    for _ in range(trotter_steps): # add the parameterised trotter step a repeated number of times
        qc.append(trotterisation, [quantum_register[index] for index in target_qubit_indices]) 

    if trotter_steps > 0:
        return qc.bind_parameters({t: time_slice}) # assign the parameter
    else:
        return qc # do nothing and return input circuit 

def build_trot_qc():
    '''Constructs the trotterisation used in our submission

    returns 
        QuantumCircuit
    '''
    # Build a subcircuit for XX(t) two-qubit gate
    XX_qr = QuantumRegister(2)
    XX_qc = QuantumCircuit(XX_qr, name='XX')

    XX_qc.ry(np.pi/2,[0,1])
    XX_qc.cnot(0,1)
    XX_qc.rz(2 * t, 1)
    XX_qc.cnot(0,1)
    XX_qc.ry(-np.pi/2,[0,1])

    # Convert custom quantum circuit into a gate
    XX = XX_qc.to_instruction()

    # Build a subcircuit for YY(t) two-qubit gate
    YY_qr = QuantumRegister(2)
    YY_qc = QuantumCircuit(YY_qr, name='YY')

    YY_qc.rx(np.pi/2,[0,1])
    YY_qc.cnot(0,1)
    YY_qc.rz(2 * t, 1)
    YY_qc.cnot(0,1)
    YY_qc.rx(-np.pi/2,[0,1])

    # Convert custom quantum circuit into a gate
    YY = YY_qc.to_instruction()

    # Build a subcircuit for ZZ(t) two-qubit gate
    ZZ_qr = QuantumRegister(2)
    ZZ_qc = QuantumCircuit(ZZ_qr, name='ZZ')

    ZZ_qc.cnot(0,1)
    ZZ_qc.rz(2 * t, 1)
    ZZ_qc.cnot(0,1)

    # Convert custom quantum circuit into a gate
    ZZ = ZZ_qc.to_instruction()

    # Combine subcircuits into a single multiqubit gate representing a single trotter step
    num_qubits = 3

    Trot_qr = QuantumRegister(num_qubits)
    Trot_qc = QuantumCircuit(Trot_qr, name='Trot')

    for i in range(0, num_qubits - 1):
        Trot_qc.append(ZZ, [Trot_qr[i], Trot_qr[i+1]])
        Trot_qc.append(YY, [Trot_qr[i], Trot_qr[i+1]])
        Trot_qc.append(XX, [Trot_qr[i], Trot_qr[i+1]])
    
    return Trot_qc

# Convert custom quantum circuit into a gate
Trot_gate = build_trot_qc().to_instruction()


def get_approximate_final_time_stomo_circuits(target_time, trotter_steps, num_qubits, backend, shots, reps, isl, sufficient_cost=1e-2):
    '''Get state tomography circuits without executing - for use when jobs are being retrieved

    Args: 
        target_time (float) : Final evolution time.
        trotter_steps (int) : Number of trotter steps used in simulation.
        num_qubits (int) : the number of qubits in the circuit
        backend (BaseBackend or Backend) : Backend to execute circuits on.
        shots (int) : Number of repetitions of each circuit, for sampling
        reps (int) : Number of repetitions for each circuit, for obtaining statistics.
        isl (bool) : True : use ISl recompiler, False : do not recompile.
        sufficient_cost (float) : Termination condition for recompilation. Default 1e-2. 

    Returns:
        List[QuantumCircuits]
    '''
    target_qubit_indices = [1,3,5]
    
    quantum_register = QuantumRegister(num_qubits)
    qc = QuantumCircuit(quantum_register)
    # state prep of |110> state (little endian)
    qc.x([3,5])

    time_slice = target_time / trotter_steps if trotter_steps > 0 else 0

    qc = build_trotter_circuit(qc, quantum_register, target_qubit_indices, Trot_gate, trotter_steps, time_slice)

    if isl: # recompile the circuit
        fname = get_isl_recompiled_partial_trotter_circuit_filename(round(target_time,4), trotter_steps, trotter_steps, num_qubits, sufficient_cost)
        recompiled_qc = ISL(qc, fname, sufficient_cost)['circuit']
    else:
        recompiled_qc = qc # don't recompile
    
    return state_tomography_circuits(recompiled_qc, target_qubit_indices) # return the circuits required for state tomography 

