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
    """
    This function computes the coefficients alpha_i according to the formulas presented in Ref. [5]. 
    For a given unitary and an index set, it will return the sum over all the polynomials consisting 
    of the contributing unitary matrix elements v_{sr} for a given transition between input- and output states (this is specified by the input index).
    """

    input_length = int(len(index) / 2)

    # Divide index set into input and output state
    input_mode_occupations, output_mode_occupations = index[0:input_length] + [1,1,1], index[input_length:] + [1,1,1]

    # Check for particle number conservation
    n_input = sum(input_mode_occupations)
    n_output = sum(output_mode_occupations)

    if n_input != n_output:
        return 0

    # List of the occupied input mode, this is important to correctly compute the operator P from Ref [5].
    occupied_input_modes = [index for index, occupation in enumerate(input_mode_occupations) if occupation==1]

    # This list stores which creation operator has to be applied how many times to create the desired output state.
    creation_ops = []
    for mode, occupation in enumerate(output_mode_occupations):
        for _ in range(occupation):
            creation_ops.append(mode)

    # Then, all permutations of valid combinations of creation operators are computed.
    permutations = list(itertools.permutations(creation_ops))

    # These permutations are then used to compute all the polynomials in v_{sr} to make up the unitary matrix.
    alpha = 0

    for permutation in permutations:
        poly = 1
        for index, mode in enumerate(occupied_input_modes):
            poly *= unitary[mode, permutation[index]]

        alpha += poly
    
    return alpha


def partition_min_max(n, k, l, m):
    """
    This method provides an integer partitition of an integer n into k integers (including 0). 
    We use this to compute all possible output states respecting particle number conservation.

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
    """
    This is the method that actually computes all the permutations of the partitions. 

    n: The integer to partition
    k: The length of partitions
    """
    partitions = partition_min_max(n, k, 0, n)

    permutations = []
    for partition in partitions:
        permutations.extend(list(itertools.permutations(partition)))
    
    return map(list, list(set(permutations)))


def loss_function_ccz_dual_rail(U):
    """
    This is the loss function to optimize the CCZ gate using dual rail encoding. 
    The desired gate loss makes sure that all desired amplitudes are equal (up to a sign for the |111> state of course).
    The undesired gate loss on the other hand makes sure that all undesired coefficients vanish. 

    The conditions for these coefficients alpha were once again derived following Ref. [5], but with 3 heralding modes instead (according to Ref. [6]). 
    """
    desired_gate_loss = 0
    + np.abs(get_alpha([0,0,0,0,0,0], U) - get_alpha([0,1,1,0,1,1], U))**2 
    + np.abs(get_alpha([0,1,1,0,1,1], U) - get_alpha([1,0,1,1,0,1], U))**2
    + np.abs(get_alpha([1,0,1,1,0,1], U) - get_alpha([1,1,0,1,1,0], U))**2
    + np.abs(get_alpha([1,1,0,1,1,0], U) - get_alpha([0,0,1,0,0,1], U))**2
    + np.abs(get_alpha([0,0,1,0,0,1], U) - get_alpha([0,1,0,0,1,0], U))**2
    + np.abs(get_alpha([0,1,0,0,1,0], U) - get_alpha([1,0,0,1,0,0], U))**2
    + np.abs(get_alpha([1,0,0,1,0,0], U) + get_alpha([1,1,1,1,1,1], U))**2

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

def loss_function_ccz_hybrid(U):
    """
    This is the loss function to optimize the CCZ gate using the hybrid encoding. 
    The desired gate loss makes sure that all desired amplitudes are equal (up to a sign).
    The undesired gate loss on the other hand makes sure that all undesired coefficients vanish. 

    The conditions for these coefficients alpha were once again derived following Ref. [5], but with 3 heralding modes instead (according to Ref. [6]). 
    """

    desired_gate_loss = 0
    + np.abs(get_alpha([1,0,0,0,1,0,1,0,0,0,1,0], U) - get_alpha([1,0,0,0,0,1,1,0,0,0,0,1], U))**2 
    + np.abs(get_alpha([1,0,0,0,0,1,1,0,0,0,0,1], U) - get_alpha([0,1,0,0,1,0,0,1,0,0,1,0], U))**2
    + np.abs(get_alpha([0,1,0,0,1,0,0,1,0,0,1,0], U) - get_alpha([0,1,0,0,0,1,0,1,0,0,0,1], U))**2
    + np.abs(get_alpha([0,1,0,0,0,1,0,1,0,0,0,1], U) - get_alpha([0,0,1,0,1,0,0,0,1,0,1,0], U))**2
    + np.abs(get_alpha([0,0,1,0,1,0,0,0,1,0,1,0], U) - get_alpha([0,0,1,0,0,1,0,0,1,0,0,1], U))**2
    + np.abs(get_alpha([0,0,1,0,0,1,0,0,1,0,0,1], U) - get_alpha([0,0,0,1,1,0,0,0,0,1,1,0], U))**2
    + np.abs(get_alpha([0,0,0,1,1,0,0,0,0,1,1,0], U) + get_alpha([0,0,0,1,0,1,0,0,0,1,0,1], U))**2

    undesired_gate_loss = 0

    for input_state in [[1,0,0,0,1,0], [1,0,0,0,0,1], [0,1,0,0,1,0], [0,1,0,0,0,1], [0,0,1,0,1,0], [0,0,1,0,0,1], [0,0,0,1,1,0], [0,0,0,1,0,1]]:
        
        particle_number = np.sum(input_state)

        output_states = get_partitions_permutations(n=particle_number, k=6)

        for output_state in output_states:
            if input_state != output_state:
                
                index=input_state + list(output_state) 
                undesired_gate_loss += np.abs(get_alpha(index=index, unitary=U))**2

    loss = desired_gate_loss + undesired_gate_loss

    return loss

def loss_function_toffoli_hybrid(U):
    """
    This is the loss function to optimize the Toffoli gate using the hybrid encoding. 
    The desired gate loss makes sure that all desired amplitudes are equal.
    The undesired gate loss on the other hand makes sure that all undesired coefficients vanish. 

    The conditions for these coefficients alpha were once again derived following Ref. [5], but with 3 heralding modes instead. 
    The last two lines of the desired gate loss reflect that the Toffoli gate is non-diagonal.
    """

    desired_gate_loss = 0
    + np.abs(get_alpha([1,0,0,0,1,0,1,0,0,0,1,0], U) - get_alpha([1,0,0,0,0,1,1,0,0,0,0,1], U))**2 
    + np.abs(get_alpha([1,0,0,0,0,1,1,0,0,0,0,1], U) - get_alpha([0,1,0,0,1,0,0,1,0,0,1,0], U))**2
    + np.abs(get_alpha([0,1,0,0,1,0,0,1,0,0,1,0], U) - get_alpha([0,1,0,0,0,1,0,1,0,0,0,1], U))**2
    + np.abs(get_alpha([0,1,0,0,0,1,0,1,0,0,0,1], U) - get_alpha([0,0,1,0,1,0,0,0,1,0,1,0], U))**2
    + np.abs(get_alpha([0,0,1,0,1,0,0,0,1,0,1,0], U) - get_alpha([0,0,1,0,0,1,0,0,1,0,0,1], U))**2
    + np.abs(get_alpha([0,0,1,0,0,1,0,0,1,0,0,1], U) - get_alpha([0,0,0,1,1,0,0,0,0,1,0,1], U))**2
    + np.abs(get_alpha([0,0,0,1,1,0,0,0,0,1,0,1], U) - get_alpha([0,0,0,1,0,1,0,0,0,1,1,0], U))**2

    undesired_gate_loss = 0

    for input_state in [[1,0,0,0,1,0], [1,0,0,0,0,1], [0,1,0,0,1,0], [0,1,0,0,0,1], [0,0,1,0,1,0], [0,0,1,0,0,1], [0,0,0,1,1,0], [0,0,0,1,0,1]]:
        
        particle_number = np.sum(input_state)

        output_states = get_partitions_permutations(n=particle_number, k=6)

        if input_state == [0,0,0,1,1,0]:

            for output_state in output_states:

                if [0,0,0,1,0,1] != output_state:
                    
                    index=input_state + list(output_state) 
                    undesired_gate_loss += np.abs(get_alpha(index=index, unitary=U))**2

        elif input_state == [0,0,0,1,0,1]:
            
            for output_state in output_states:

                if [0,0,0,1,1,0] != output_state:
                    
                    index=input_state + list(output_state) 
                    undesired_gate_loss += np.abs(get_alpha(index=index, unitary=U))**2

        else:
            for output_state in output_states:

                if input_state != output_state:
                    
                    index=input_state + list(output_state) 
                    undesired_gate_loss += np.abs(get_alpha(index=index, unitary=U))**2

    loss = desired_gate_loss + undesired_gate_loss

    return loss

def loss_function_bonus_hybrid(U):
    """
    This is the loss function to optimize the bonus gate (CZ12 CZ23) using the hybrid encoding. 
    The desired gate loss makes sure that all desired amplitudes are equal.
    The undesired gate loss on the other hand makes sure that all undesired coefficients vanish. 

    The conditions for these coefficients alpha were once again derived following Ref. [5], but with 3 heralding modes instead. 
    """

    desired_gate_loss = 0
    + np.abs(get_alpha([0,0,0,0,0,0], U) - get_alpha([0,0,1,0,0,1], U))**2 
    + np.abs(get_alpha([0,0,1,0,0,1], U) - get_alpha([0,1,0,0,1,0], U))**2
    + np.abs(get_alpha([0,1,0,0,1,0], U) + get_alpha([0,1,1,0,1,1], U))**2
    + np.abs(get_alpha([0,1,1,0,1,1], U) + get_alpha([1,0,0,1,0,0], U))**2
    + np.abs(get_alpha([1,0,0,1,0,0], U) + get_alpha([1,0,1,1,0,1], U))**2
    + np.abs(get_alpha([1,0,1,1,0,1], U) + get_alpha([1,1,0,1,1,0], U))**2
    + np.abs(get_alpha([1,1,0,1,1,0], U) - get_alpha([1,1,1,1,1,1], U))**2

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
    """
    Computes the success probability for both the CCZ gate in dual rail encoding and the bonus gate in hybrid encoding.
    The formula is according to Ref. [5].
    """
    return np.abs(get_alpha(index=[0,0,0,0,0,0], unitary=U))**2


def get_success_prob_for_hybrid_toffoli_ccz(U):
    """
    Computes the success probability for both the CCZ- and Toffoli gate in hybrid encoding.
    The formula is according to Ref. [5].
    """
    return np.abs(get_alpha(index=[0,0,0,0,0,0,0,0,0,0,0,0], unitary=U))**2

def get_CCZ_unitary():

    best_loss = np.infty
    best_prob = 0
    best_U = None

    timer = time.time()

    try:
        for _ in range(1000):

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


def embed_submatrix(U):
    """
    This method takes the unitary submatrix computed for the CCZ gate in dual rail encoding and the bonus gate in hybrid encoding (acting on 6 modes) 
    and embeds it into the space consisting of all 9 modes.
    """

    # Create full-space identity matrix
    A = np.identity(9, dtype=np.complex128)

    # Specify the rows and columns where we want to insert the submatrix
    indices_to_insert = [1, 3, 5, 6, 7, 8]  

    # Insert submatrix into A
    for i, row in enumerate(indices_to_insert):
        for j, col in enumerate(indices_to_insert):
            A[row, col] = U[i, j]

    return A


def get_CCZ():

    unitary = get_CCZ_unitary()
    unitary = embed_submatrix(unitary)
    M = pcvl.Matrix(unitary)
    Unitary_matrix = comp.Unitary(U=M)

    mzi = comp.BS() // (0, comp.PS(pcvl.Parameter("phi1"))) // comp.BS() // (0, comp.PS(pcvl.Parameter("phi2")))

    circuit = pcvl.Circuit.decomposition(M, mzi, shape="triangle")
    circuit.describe()

    p = pcvl.Processor("Naive", circuit)
    p.add_herald(6,1)  # Third mode is heralded (1 photon in, 1 photon expected out)
    p.add_herald(7,1)  # Fourth mode is heralded (1 photon in, 1 photon expected out)
    p.add_herald(8,1)  # Fifth mode is heralded (1 photon in, 1 photon expected out)

    # pcvl.pdisplay(p, recursive=True)

    return p


# get_CCZ()