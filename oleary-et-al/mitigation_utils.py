from qiskit import QuantumRegister, execute
from qiskit.ignis.mitigation.measurement import *



def do_measurement_calibration(backend, num_qubits, target_qubit_indices, cal_results=None):
    ''' Wrapper function for measurement calibration.

        Args:
             backend (BaseBackend or Backend) : Backend to execute circuits on.
             num_qubits (int) : the number of qubits in the circuit
             target_qubit_indices (List[int]) : List of qubits to calibrate
             cal_results (Result) : Option to load existing measurement calibration data to rebuild the fitter.

        Returns
            CompleteMeasFitter
    '''   
    qr = QuantumRegister(num_qubits)
    meas_calibs, state_labels = complete_meas_cal(qubit_list=target_qubit_indices, qr=qr, circlabel='mcal')

    if cal_results is None: # if no preexisting result used, run circuits

        job = execute(meas_calibs, backend, shots=5000, optimization_level=3)

        cal_results = job.result() 
            
    return CompleteMeasFitter(cal_results, state_labels, qubit_list=None, circlabel='mcal')


