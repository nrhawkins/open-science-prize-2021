from qiskit import execute
from qiskit.tools.monitor import job_monitor


def build_job_list(qcs, backend, shots, reps):
    '''Execute multiple circuits indivudally a repeated number of times.

        Args
            qcs (List[QuantumCircuit]) : list of quantum circuits to run
            backend (BaseBackend or Backend) : Backend to execute circuits on.
            shots (int) : Number of repetitions of each circuit, for sampling
            reps (int) : Number of repetitions for each circuit, for obtaining statistics.
    '''
    jobs = []
    
    for _ in range(reps): # execute and collect jobs for each rep
        # execute
        job = execute(qcs, backend, shots=shots)
        print('Job ID', job.job_id())
        jobs.append(job)
        
    return jobs


def build_batch_job_list(qcs, backend, shots):
    ''' Args
            qcs (List[QuantumCircuit]) : list of quantum circuits to run
            backend (BaseBackend or Backend) : Backend to execute circuits on.
            shots (int) : Number of repetitions of each circuit, for sampling
    '''
    return execute(qcs, backend, shots=shots)


def monitor(job_list):
    ''' Args
            job_list (List[BaseJob]) : list of jobs to monitor.
    '''
    for job in job_list:
        job_monitor(job) # monitor each job
        try:
            if job.error_message() is not None:
                print(job.error_message()) # print errors if they appear
        except:
            pass

