
import perceval as pcvl
from perceval.components import catalog
from perceval.converters import QiskitConverter, MyQLMConverter
from perceval.algorithm import Analyzer, Sampler

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

import qat.lang.AQASM as qataqasm

print('all imports set')

if 0: # a cz gate
    qcircuit = QuantumCircuit(2)
    qcircuit.h(1)
    qcircuit.cx(0, 1)
    qcircuit.h(1)
    #qcircuit.draw()

    qiskit_converter = QiskitConverter(catalog, backend_name="Naive")
    quantum_processor = qiskit_converter.convert(qcircuit, use_postselection=True)
    pcvl.pdisplay(quantum_processor, recursive=True)
    print('to Perceval')

if 0: # a ccz gate, to purposely get an error.
    qcircuit = QuantumCircuit(3)
    qcircuit.h(2)
    qcircuit.ccx(0, 1, 2)
    qcircuit.h(2)
    #qcircuit.draw()

    qiskit_converter = QiskitConverter(catalog, backend_name="Naive")
    quantum_processor = qiskit_converter.convert(qcircuit, use_postselection=True)
    pcvl.pdisplay(quantum_processor, recursive=True)
    print('to Perceval')
    #ValueError: Gates with number of Qubits higher than 2 not implemented
