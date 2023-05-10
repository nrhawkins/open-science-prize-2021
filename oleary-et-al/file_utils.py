'''
    A convenience script for all the file names used to retrieve past results, designed so the particular experiment parameters can uniquely define
    a chosen execution.

    All results here are related to the trotterised evolution of the |110> computational basis state under the Heisenberg XXX Hamiltonian. 
'''

# directories

def get_data_dir():
    return 'ibm-qsim-challenge-data/'

def get_circuits_dir():
    return 'ibm-qsim-challenge-isl-circuits/'

def get_meas_cal_dir():
    return 'ibm-qsim-challenge-meas-cal/'

'''
    Partial trotter circuit : for some timeslice (or timestep) defined by: final_time / max_trotter_steps, a 'partial trotter circuit' is a
    circuit comprised of some number of repetitions of a trotter circuit, parameterised by the timeslice, where the number of repetitions is 
    less than the number: max_trotter_steps. 
'''

# state probabilities

def get_partial_trotter_circuit_state_probs_filename(start_time, final_time, max_trotter_steps, num_trotter_steps, backend_name, shots, reps):
    '''state probabilities after running unrecompiled partial trotter circuit'''
    return get_data_dir() + 'state_probs_at_each_trotter_step_time_{}_{}_maxtrotsteps_{}_numtrotsteps_{}_backend_{}_shots_{}_reps_{}'.format(start_time,      final_time, max_trotter_steps, num_trotter_steps, backend_name, shots, reps)

def get_partial_trotter_circuit_state_probs_isl_filename(start_time, final_time, max_trotter_steps, num_trotter_steps, backend_name, shots, reps,             sufficient_cost):
    '''state probabilities after running recompiled partial trotter circuit'''
    return get_data_dir() + 'state_probs_at_each_trotter_step_isl_time_{}_{}_maxtrotsteps_{}_numtrotsteps_{}_backend_{}_shots_{}_reps_{}_suffcost_{}'.        format(start_time, final_time, max_trotter_steps, num_trotter_steps, backend_name, shots, reps, sufficient_cost)

def get_full_trotter_circuit_results_isl_meas_cal_filename(start_time, final_time, max_trotter_steps, backend_name, shots, reps, sufficient_cost):
    '''state probabilities after running recompiled partial trotter circuit with interpolated measurement calibration.'''
    return get_data_dir() + 'qiskit_results_trotter_circuits_isl_meas_cal_time_{}_{}_maxtrotsteps_{}_backend_{}_shots_{}_reps_{}_suffcost_{}'.                format(start_time, final_time, max_trotter_steps, backend_name, shots, reps, sufficient_cost)

# state tomography result objects and circuits

def get_partial_trotter_stomo_circuits_results_circuits_isl_filename(start_time, final_time, max_trotter_steps, num_trotter_steps, backend_name, shots, reps, sufficient_cost):
    '''qiskit result objects and state tomography circuits obtained after executing full state tomography circuits on a recompiled partial trotter circuit.'''
    return get_data_dir() +                                                                                                                                   'qiskit_stomo_results_circuits_at_each_trotter_step_isl_time_{}_{}_maxtrotsteps_{}_numtrotsteps_{}_backend_{}_shots_{}_reps_{}_suffcost_{}'.                  format(start_time, final_time, max_trotter_steps, num_trotter_steps, backend_name, shots, reps, sufficient_cost)

def get_partial_trotter_stomo_circuits_results_circuits_meas_cal_filename(start_time, final_time, max_trotter_steps, num_trotter_steps, backend_name, shots, reps):
    '''qiskit result objects and state tomography circuits obtained after executing full state tomography circuits on a partial trotter circuit. Includes 
        measurement calibration results interpolated between the state tomography results.
    '''
    return get_data_dir() +                                                                                                                                   'qiskit_stomo_results_circuits_at_each_trotter_step_meas_cal_time_{}_{}_maxtrotsteps_{}_numtrotsteps_{}_backend_{}_shots_{}_reps_{}'.                  format(start_time, final_time, max_trotter_steps, num_trotter_steps, backend_name, shots, reps)

def get_partial_trotter_stomo_circuits_results_circuits_filename(start_time, final_time, max_trotter_steps, num_trotter_steps, backend_name, shots, reps):
    '''qiskit result objects and state tomography circuits after executing full state tomography circuits on an unrecompiled partial trotter circuit.'''
    return get_data_dir() +                                                                                                                                   'qiskit_stomo_results_circuits_at_each_trotter_step_time_{}_{}_maxtrotsteps_{}_numtrotsteps_{}_backend_{}_shots_{}_reps_{}'.                  format(start_time, final_time, max_trotter_steps, num_trotter_steps, backend_name, shots, reps)

# cached ISL result objects

def get_isl_recompiled_partial_trotter_circuit_filename(target_time, max_trotter_steps, trotter_step, num_qubits, sufficient_cost):
    '''cached ISL results object for a partial trotter circuit and a given sufficient cost.'''
    return get_circuits_dir() + 'isl_partial_trot_circ_targettime_{}_maxtrotstep_{}_trotstep_{}_nq_{}_suffcost_{}'.format(target_time, max_trotter_steps,     trotter_step, num_qubits, sufficient_cost)

