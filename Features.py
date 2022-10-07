# - (Shannon) Entropy values; entropy values can be taken as a measure of complexity of the signal.
# - Statistical features like:
#   * Variance
#   * Standard deviation
#   * Mean
#   * Median
#   * 25th percentile value
#   * 75th percentile value
#   * Root Mean Square value; square of the average of the squared amplitude values
# - The mean of the derivative
#   * Zero crossing rate, i.e. the number of times a signal crosses y = 0
#   * Mean crossing rate, i.e. the number of times a signal crosses y = mean(y)

# Μοre information can be found in https://ataspinar.com/2018/12/21/a-guide-for-using-the-wavelet-transform-in-machine-learning/



from collections import Counter
import scipy
import numpy as np
# ['Entropy', 'N5', 'N25', 'N75', 'N95', 'Median', 'Mean', 'Std', 'Var', 'RMS', 'Zero_Crossings_rate', 'Mean_Crossing_rate']

def calculate_entropy(list_values):
    
    counter_values = Counter(list_values).most_common()
    probabilities  = [elem[1]/len(list_values) for elem in counter_values]
    entropy        = scipy.stats.entropy(probabilities)
    
    return entropy
 
def calculate_statistics(list_values):
    n5     = np.nanpercentile(list_values, 5)
    n25    = np.nanpercentile(list_values, 25)
    n75    = np.nanpercentile(list_values, 75)
    n95    = np.nanpercentile(list_values, 95)
    median = np.nanpercentile(list_values, 50)
    mean   = np.nanmean(list_values)
    std    = np.nanstd(list_values)
    var    = np.nanvar(list_values)
    rms    = np.nanmean(np.sqrt(list_values**2))
    
#     return [n5, n25, n75, n95, median, mean, std, var, rms]
    return [n25, n75, median, mean, std, var, rms]
 

def calculate_crossings(list_values):
    zero_crossing_indices = np.nonzero(np.diff(np.array(list_values) > 0))[0]
    no_zero_crossings     = len(zero_crossing_indices)
    mean_crossing_indices = np.nonzero(np.diff(np.array(list_values) > np.nanmean(list_values)))[0]
    no_mean_crossings     = len(mean_crossing_indices)
    
    return [no_zero_crossings, no_mean_crossings]
 
    
def get_features(list_values):
    entropy    = calculate_entropy(list_values)
    statistics = calculate_statistics(list_values)
    crossings  = calculate_crossings(list_values)
    
    return [entropy] + crossings + statistics

