from qiskit.ignis.verification.tomography import StateTomographyFitter
from qiskit.quantum_info import state_fidelity

from qiskit.opflow import Zero, One

import numpy as np

from misc_utils import *

def state_tomo(result, st_qcs):
    ''' Determines fidelity of state generated in result with the |110> state.

        Args:
            result (Result) : result of a quantum circuit execution.
            st_qcs (List[QuantumCircuit]) : a list of state tomography circuits used to obtain the results.

        Returns:
            Float
    '''
    # The expected final state; necessary to determine state tomography fidelity
    target_state = (One^One^Zero).to_matrix()  # DO NOT MODIFY (|q_5,q_3,q_1> = |110>)
    # Fit state tomography results
    tomo_fitter = StateTomographyFitter(result, st_qcs)
    rho_fit = tomo_fitter.fit(method='lstsq')
    # Compute fidelity
    fid = state_fidelity(rho_fit, target_state)
    return fid

def fidelities_from_st_tomo_circuits(jobs_for_each_trotter_step, st_qcs_per_timestep):
    ''' Determines fidelities for multiple jobs

        Args:
            jobs_for_each_trotter_step (List[BaseJob]) : Jobs for multiple circuit executions
            st_qcs_per_timestep : State tomography circuits used in each execution.

        Returns:
            List[Floats]

    '''
    fids = []

    for jobs_stqc in zip(jobs_for_each_trotter_step, st_qcs_per_timestep): # pair the jobs and circuits together
        fids_for_each_trotter_step = []
        for job in jobs_stqc[0]: # for each rep
            fid = state_tomo(job.result(), jobs_stqc[1])
            fids_for_each_trotter_step.append(fid)
        fids.append(np.mean(fids_for_each_trotter_step)) # keep the average fidelity

    return fids


def state_probabilities_from_circuit_multi_reps(jobs):
    ''' Combines the results from doing some number of repetitions of a quantum circuit execution.
        Returns an dictionary of averaged state probabilities.

        Args
            jobs (List[BaseJob]) : a set of jobs resulting from running the same circuit some number of times.

        Returns:
            Dict[string : float]
    '''
    total_state_probs = {}

    for job in jobs: # for each rep
        result = job.result()
        total_state_probs = dict_sum(total_state_probs, counts_to_probabilities(result.get_counts())) # convert to probabilities and to the output dictionary

    return { k : (total_state_probs[k] / len(jobs)) for k in total_state_probs} # return the average over the reps

def counts_to_probabilities(counts):
    ''' Converts counts dictionary to state probability dictionary.

        Args:
            counts (Dict[string, float]) : counts dictionary of a quantum circuit execution

        Returns:
            Dict[string, float]
    '''
    total_counts = sum([counts[k] for k in counts])
    return {k : counts[k] / total_counts for k in counts}


def get_state_probs_at_each_trotter_step(all_counts, reps):
    ''' Takes the results of running trotter step circuits of increasing depth to approximate
        time evolution at increasing timesteps, and returns state probabities for each computational
        basis state at each of these timesteps. Method basically averages over multiple reps.

        Args:
            all_counts (List[Dict[string, float]] : a list of count dictionaries, intended to be of the
            form: [counts_trotter_step_i for _ in range(reps_] + [counts_trotter_step_i+1 for _ in range(reps_] + ...

        Returns:
            List[Dict[string, float]]
    '''
    state_probs_at_each_trotter_step = []

    for i in range(0, len(all_counts), reps): # 
        total_state_probs = {}

        for counts in all_counts[i : i + reps]: # i.e. for each rep
            total_state_probs = dict_sum(total_state_probs, counts_to_probabilities(counts)) # convert to probabilities and add to output dictionary.

        total_state_probs = { k : (total_state_probs[k] / reps) for k in total_state_probs} # average over the reps
        state_probs_at_each_trotter_step.append(total_state_probs)  

    return state_probs_at_each_trotter_step

