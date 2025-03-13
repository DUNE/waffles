# -------- Plot a function in an APA grid --------

import waffles.plotting.drawing_tools as draw
wset = draw.read("../data/wfset_10_30201.hdf5",0,1)


# -------- Time offset histograms --------
draw.plot_function_grid(wset, apa=2,tmin=-1000,tmax=1000,  plot_function=draw.plot_to_function)
input()

# ----------- Sigma vs TS plots -----------
draw.plot_function_grid(wset, apa=2,tmin=-1000,tmax=1000,  plot_function=draw.plot_sigma_vs_tsfunction)
input()

# ----------- Sigma histograms -----------
draw.plot_function_grid(wset, apa=2,tmin=-1000,tmax=1000,  plot_function=draw.plot_sigma_function)
