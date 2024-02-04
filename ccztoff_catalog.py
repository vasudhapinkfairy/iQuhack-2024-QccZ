

import math
import perceval as pcvl
from tqdm.auto import tqdm
import numpy as np
import matplotlib

#import perceval.components.unitary_components as comp
#from perceval.components import BS, PERM

print(pcvl.catalog.list())
if 1:
    czgate = pcvl.catalog['heralded cz'].build_processor() 
    pcvl.pdisplay(czgate, recursive=True) #saved image
if 0:
    the_toffoli = pcvl.catalog['toffoli'].build_processor() 
    pcvl.pdisplay(the_toffoli, recursive=True) #saved image

states = {
    pcvl.BasicState([1, 0, 1, 0]): "00",
    pcvl.BasicState([1, 0, 0, 1]): "01",
    pcvl.BasicState([0, 1, 1, 0]): "10",
    pcvl.BasicState([0, 1, 0, 1]): "11"
}

ca = pcvl.algorithm.Analyzer(czgate, states)
truth_table= {'00':'00', '01':'01', '10':'10', '11':'-11'}
ca.compute(expected=truth_table)
pcvl.pdisplay(ca)
print(f"performance = {ca.performance}, fidelity = {ca.fidelity*100}%")







