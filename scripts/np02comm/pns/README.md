## How to use...
The jupyter monitoring can process many runs. \
Because cathode data is fullstream and takes lots memory, we only processing each file one by one. \
At the end, all peaks found are merged and written in a dictionary where it can be plotted at the end. \
The peak finder is done by:
- Apply first derivative
- Apply two moving average to smooth the derivative
- Apply second derivative
- Find peaks in the second derivative by and select by amplitude of first derivative. 


Peaks > 1000 give a "dead time" of 1000 ticks. \
After finding a peak, add a "dead time" of 100 ticks to avoid finding group of signals all together.
