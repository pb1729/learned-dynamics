import sys
import re
import numpy as np


# This program neatly formats print statements in Triton kernels back into tensors.
# Expected print statement format:
# print("[ %f ]", tensor_to_print)
# separate print statements in loops by putting print("next") inside the loop, at the end of the loop body
# Command line args are tensor dims. eg. 16 16 for a tensor with shape (16, 16). Undersizing dims is ok.

# constants
LINEWIDTH = 160

np.set_printoptions(linewidth=LINEWIDTH, threshold=64**2 + 1, formatter={"float": lambda x: f"{x: >+07.2f}"})

def get_re(thread_dims):
    num_group = r"\ *([0-9]*)"
    flt_group = r"\ *([0-9e\-\.]*)\ *"
    return (
        r"pid\ \(" + num_group + r"\," + num_group + r"\," + num_group + r"\)\ "
        + r"idx\ \(" + r"\,".join([num_group]*thread_dims) + r"\)\ "
        + r"\[" + flt_group + r"\]")

shape = [int(arg) for arg in sys.argv[1:]]
pattern = get_re(len(shape))

while True:
    indices = []
    values = []
    for line in sys.stdin:
        m = re.search(pattern, line)
        if m is not None:
            recorded_data = [s for s in m.groups()]
            idx = tuple([int(s) for s in recorded_data[:3 + len(shape)]])
            val = float(recorded_data[3 + len(shape)])
            if all([idx[3 + j] < shape[j] for j in range(len(shape))]):
                indices.append(idx)
                values.append(val)
            continue
        m = re.search(r"next", line) # you can print the "next" keyword to delimit for loops, etc.
        if m is not None: break
        # otherwise, we just print the line as normal
        print(line, end="")
    if len(indices) == 0: continue # ignore rounds where nothing was recorded

    grid_0 = 1 + max([idx[0] for idx in indices])
    grid_1 = 1 + max([idx[1] for idx in indices])
    grid_2 = 1 + max([idx[2] for idx in indices])

    ans = np.zeros((grid_0, grid_1, grid_2, *shape))

    for idx, val in zip(indices, values):
        ans[idx] = val

    print(ans)
    print(ans.shape)
    print("next")


