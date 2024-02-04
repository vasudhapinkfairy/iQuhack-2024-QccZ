
'''
main.py should contain a method called "get_CCZ" with no argument that return a perceval Processor
'''

import scipy
import numpy as np
from scipy.stats import unitary_group
import itertools
import time
import math
import perceval as pcvl
from perceval.components.linear_circuit import Circuit
import perceval.components as comp


def get_alpha(index: list, unitary: np.ndarray):

    input_mode_occupations, output_mode_occupations = index[0:3] + [1,1,1], index[3:6] + [1,1,1]

    n_input = sum(input_mode_occupations)
    n_output = sum(output_mode_occupations)

    if n_input != n_output:
        return 0

    occupied_input_modes = [index for index, occupation in enumerate(input_mode_occupations) if occupation==1]

    idk_how_to_name_this = []
    for mode, occupation in enumerate(output_mode_occupations):
        for _ in range(occupation):
            idk_how_to_name_this.append(mode)

    permutations = list(itertools.permutations(idk_how_to_name_this))

    alpha = 0

    for permutation in permutations:
        poly = 1
        for index, mode in enumerate(occupied_input_modes):
            poly *= unitary[mode, permutation[index]]

        alpha += poly
    
    return alpha


def partition_min_max(n, k, l, m):
    """
    n: The integer to partition
    k: The length of partitions
    l: The minimum partition element size
    m: The maximum partition element size
    """
    if k < 1:
        return []
    if k == 1:
        if l <= n <= m:
            return [(n,)]
        return []
    result = []
    for i in range(l, m + 1):
        sub_partitions = partition_min_max(n - i, k - 1, i, m)
        for sub_partition in sub_partitions:
            result.append(sub_partition + (i,))
    return result

def get_partitions_permutations(n, k):
    partitions = partition_min_max(n, k, 0, n)

    permutations = []
    for partition in partitions:
        permutations.extend(list(itertools.permutations(partition)))
    
    return map(list, list(set(permutations)))


def loss_function_ccz_dual_rail(U):
    desired_gate_loss = 0
    + np.abs(get_alpha([0,0,0,0,0,0], U) - get_alpha([0,1,1,0,1,1], U))**2 
    + np.abs(get_alpha([0,1,1,0,1,1], U) - get_alpha([1,0,1,1,0,1], U))**2
    + np.abs(get_alpha([1,0,1,1,0,1], U) - get_alpha([1,1,0,1,1,0], U))**2
    + np.abs(get_alpha([1,1,0,1,1,0], U) + get_alpha([0,0,1,0,0,1], U))**2
    + np.abs(get_alpha([0,0,1,0,0,1], U) - get_alpha([0,1,0,0,1,0], U))**2
    + np.abs(get_alpha([0,1,0,0,1,0], U) - get_alpha([1,0,0,1,0,0], U))**2
    + np.abs(get_alpha([1,0,0,1,0,0], U) - get_alpha([1,1,1,1,1,1], U))**2

    undesired_gate_loss = 0

    for input_state in [[0,0,0], [0,0,1], [0,1,0], [0,1,1], [1,0,0], [1,0,1], [1,1,0], [1,1,1]]:
        
        particle_number = np.sum(input_state)

        output_states = get_partitions_permutations(n=particle_number, k=3)

        for output_state in output_states:
            if input_state != output_state:
                
                index=input_state + list(output_state) 
                undesired_gate_loss += np.abs(get_alpha(index=index, unitary=U))**2

    loss = desired_gate_loss + undesired_gate_loss

    return loss


def get_success_prob(U):
    return np.abs(get_alpha(index=[0,0,0,0,0,0], unitary=U))**2


def get_CCZ_unitary():

    best_loss = np.infty
    best_prob = 0
    best_U = None

    timer = time.time()

    try:
        for _ in range(100):

            U = unitary_group.rvs(6)

            loss = loss_function_ccz_dual_rail(U=U)
            prob = get_success_prob(U)

            if loss < best_loss:
                best_loss = loss
                best_prob = prob
                best_U = U

        print("Calculated in", time.time()-timer, "seconds")
        print(best_loss, best_prob)
        print(U)
    except KeyboardInterrupt:
        print("Calculated in", time.time()-timer, "seconds")
        print(best_loss, best_prob)
        print(U)

    return best_U


def get_CCZ():

    unitary = get_CCZ_unitary()
    M = pcvl.Matrix(unitary)
    Unitary_matrix = comp.Unitary(U=M)

    mzi = comp.BS() // (0, comp.PS(pcvl.Parameter("phi1"))) // comp.BS() // (0, comp.PS(pcvl.Parameter("phi2")))

    circuit = pcvl.Circuit.decomposition(M, mzi, shape="triangle")
    circuit.describe()

    p = pcvl.Processor("Naive", circuit)
    p.add_herald(3,1)  # Third mode is heralded (1 photon in, 1 photon expected out)
    p.add_herald(4,1)  # Fourth mode is heralded (1 photon in, 1 photon expected out)
    p.add_herald(5,1)  # Fifth mode is heralded (1 photon in, 1 photon expected out)

    pcvl.pdisplay(p, recursive=True)

    return p