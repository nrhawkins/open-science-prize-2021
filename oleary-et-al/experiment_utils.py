from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.ignis.verification.tomography import state_tomography_circuits
from qiskit.ignis.mitigation.measurement import *

from circuit_utils import *
from file_utils import *
from job_utils import *

def simulate_heisenberg_xxx_over_time(target_time, max_trotter_steps, num_qubits, backend, shots, reps, isl, sufficient_cost=1e-2):
    '''This method constructs and executes circuits which approximate the time evolution of the |110> state under the Heisenberg XXX model from 0 
        time to target_time. Each circuit is some number of repetitions of a particular trotterisation: Trot_gate (hardcoded for the purposes of 
        the IBM QSIM Challenge, but easily generalised), starting from 0 repetitions to 'max_trotter_steps'. Each circuit is executed 'reps' number 
        of times.

        Args: 
            target_time (float) : Final evolution time.
            max_trotter_steps (int) : Max number of trotter steps used in simulation.
            num_qubits (int) : the number of qubits in the circuit
            backend (BaseBackend or Backend) : Backend to execute circuits on.
            shots (int) : Number of repetitions of each circuit, for sampling
            reps (int) : Number of repetitions for each circuit, for obtaining statistics.
            isl (bool) : True : use ISl recompiler, False : do not recompile.
            sufficient_cost (float) : Termination condition for recompilation. Default 1e-2. 

        Returns:
            List[List[BaseJob]]
    '''
    target_qubit_indices = [1,3,5] 

    jobs_for_each_trotter_step = []

    for trotter_step in range(0, max_trotter_steps + 1): # building a circuit for each timestep
        quantum_register = QuantumRegister(num_qubits)
        qc = QuantumCircuit(quantum_register)
        
        qc.x([3,5]) # state prep for |110> state (little endian)

        time_slice = target_time / max_trotter_steps if max_trotter_steps > 0 else 0

        qc = build_trotter_circuit(qc, quantum_register, target_qubit_indices, Trot_gate, trotter_step, time_slice)

        if isl: # if recompiling try and load a cached recompiled version of the circuit, or recompile from scratch if this doesn't exist
            fname = get_isl_recompiled_partial_trotter_circuit_filename(round(target_time,4), max_trotter_steps, trotter_step, num_qubits, sufficient_cost)
            recompiled_qc = ISL(qc, fname, sufficient_cost)['circuit']
        else:
            recompiled_qc = qc

        recompiled_qc.add_register(ClassicalRegister(num_qubits))
        recompiled_qc.measure(target_qubit_indices, target_qubit_indices)
        # run the circuit 'reps' number of times on the given backend, then add to the list of jobs to return
        jobs_for_each_trotter_step.append(build_job_list(recompiled_qc, backend, shots, reps))

    for jobs in jobs_for_each_trotter_step:
        monitor(jobs)

    return jobs_for_each_trotter_step


def simulate_heisenberg_xxx_over_time_interpolated_meas_cal(target_time, max_trotter_steps, num_qubits, backend, shots, reps, isl, sufficient_cost=1e-2, meas_cal=False):
    '''This method constructs and executes circuits which approximate the time evolution of the |110> state under the Heisenberg XXX model from 0 
        time to target_time. Each circuit is some number of repetitions of a particular trotterisation: Trot_gate (hardcoded for the purposes of 
        the IBM QSIM Challenge, but easily generalised), starting from 0 repetitions to 'max_trotter_steps'. Each circuit is executed 'reps' number 
        of times.

        In order to perform measurement calibration as close to orignal execution time of each circuit, here we collect the circuit for each trotter
        step in a list, then interpolate the 8 circuits required for measurement calibration into this list.

        Args: 
            target_time (float) : Final evolution time.
            max_trotter_steps (int) : Max number of trotter steps used in simulation.
            num_qubits (int) : the number of qubits in the circuit
            backend (BaseBackend or Backend) : Backend to execute circuits on.
            shots (int) : Number of repetitions of each circuit, for sampling
            reps (int) : Number of repetitions for each circuit, for obtaining statistics.
            isl (bool) : True : use ISl recompiler, False : do not recompile.
            sufficient_cost (float) : Termination condition for recompilation. Default 1e-2. 
            meas_cal (bool) : whether or not to interpolate the trotter circuits with measurement calibration circuits.

        Returns:
            List[BaseJob]
    '''
    target_qubit_indices = [1,3,5]

    circuits_for_each_trotter_step = []

    if meas_cal:
        qr = QuantumRegister(num_qubits)
        meas_calibs, state_labels = complete_meas_cal(qubit_list=target_qubit_indices, qr=qr, circlabel='mcal')

    for trotter_step in range(0, max_trotter_steps + 1): # building a circuit for each timestep
        quantum_register = QuantumRegister(num_qubits)
        qc = QuantumCircuit(quantum_register)
        
        qc.x([3,5]) # state prep for |110> state (little endian)


        time_slice = target_time / max_trotter_steps if max_trotter_steps > 0 else 0

        qc = build_trotter_circuit(qc, quantum_register, target_qubit_indices, Trot_gate, trotter_step, time_slice)

        if isl: # if recompiling try and load a cached recompiled version of the circuit, or recompile from scratch if this doesn't exist
            fname = get_isl_recompiled_partial_trotter_circuit_filename(round(target_time,4), max_trotter_steps, trotter_step, num_qubits, sufficient_cost)
            recompiled_qc = ISL(qc, fname, sufficient_cost)['circuit']
        else:
            recompiled_qc = qc

        recompiled_qc.add_register(ClassicalRegister(len(target_qubit_indices)))
        recompiled_qc.measure(target_qubit_indices, range(len(target_qubit_indices))) 

        circuits_for_each_trotter_step.append(recompiled_qc) # add the circuit to list of circuits for complete time evolution

    jobs_for_each_batch = []

    if meas_cal: # partition the circuits into 2 groups and place measurement calibration circuits between them
        for i in range(reps): # repeated for each rep
            batch_circuits = []
            batch_circuits = batch_circuits + meas_calibs
            batch_circuits = batch_circuits + circuits_for_each_trotter_step[0 : int((max_trotter_steps + 1) / 2)]
            batch_circuits = batch_circuits + meas_calibs
            batch_circuits = batch_circuits + circuits_for_each_trotter_step[int((max_trotter_steps + 1) / 2) : (max_trotter_steps + 1)]

            jobs = build_batch_job_list(batch_circuits, backend, shots) # run the batch
            jobs_for_each_batch.append(jobs)
    else: # don't partition, just run the trotter circuits, no measurement calibration.
        for i in range(reps): 
            jobs_for_each_batch = jobs_for_each_batch + build_batch_job_list(circuits_for_each_trotter_step, backend, shots)

    return jobs_for_each_batch

def simulate_heisenberg_xxx_at_fixed_time_stomo(target_time, trotter_steps, num_qubits, backend, shots, reps, isl, sufficient_cost=1e-2):
    '''This method constructs and executes circuits which approximate the time evolution of the |110> state under the Heisenberg XXX model from 0 
        time to target_time, returning only the execution results at the target time. A single trotter circuit, with 'trotter_steps' repetitions of
        the included trotterisation: 'Trot_gate' is optional recompiled then passed into the Qiskit default state_tomography circuit construction
        method, this generates 3^num_qubits circuits. 

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
            List[BaseJob]
    '''
    target_qubit_indices = [1,3,5]

    quantum_register = QuantumRegister(num_qubits)
    qc = QuantumCircuit(quantum_register)
    
    qc.x([3,5]) # state prep for |110> state (little endian)

    time_slice = target_time / trotter_steps if trotter_steps > 0 else 0

    qc = build_trotter_circuit(qc, quantum_register, target_qubit_indices, Trot_gate, trotter_steps, time_slice)

    if isl: # if recompiling try and load a cached recompiled version of the circuit, or recompile from scratch if this doesn't exist
        fname = get_isl_recompiled_partial_trotter_circuit_filename(round(target_time,4), trotter_steps, trotter_steps, num_qubits, sufficient_cost)
        recompiled_qc = ISL(qc, fname, sufficient_cost)['circuit']
    else:
        recompiled_qc = qc
    
    st_qcs = state_tomography_circuits(recompiled_qc, target_qubit_indices) # construct circuits for state tomography
    jobs = build_job_list(st_qcs, backend, shots, reps) # execute jobs

    monitor(jobs)

    return (jobs, st_qcs)

