import matplotlib.pyplot as plt
import os
import sys
import Large_Scale_Label_Propagation.func as funcs_large

# Plot parameters
plt.rcParams.update({"font.size": 30})
plt.rcParams.update({"lines.linewidth": 5})
plt.rcParams.update({"lines.markersize": 10})


path=os.path.dirname(os.getcwd())
sys.path.append(path)

labels, accuracies = funcs_large.iterative_hfs(niter=40)

plt.plot(accuracies)
plt.xlabel("Iteration")
plt.ylabel("Accuracy")
plt.title("Iterative HFS convergence")