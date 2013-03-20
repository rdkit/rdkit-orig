from scipy.stats import skew
from scipy.special import cbrt
from numpy import mean, sqrt, vstack, concatenate, hstack, array

"""
Author Jan Domanski 
Inspired by 
http://hg.adrianschreyer.eu/usrcat/src/70e075d93cd25370e7ef93301d0e28d49a0851c2/usrcat/geometry.py?at=default
from Adrian Schreyer

# Feel free to remove the above, I just want to make it clear that I have inspired
# myself on Adrian's code and give him the credit.
"""


def distances_to_point(coords, point):
    """
    Returns an array containing the distances of each coordinate in the input
    coordinates to the input point.
    """
    return sqrt(((coords-point)**2).sum(axis=1))
  
def usr_desciptor(coords):
    '''
    Calculates USR descriptor. 
    :param coords: numpy.ndarray 3N coordinate array
    :returns: tuple of 12 floats; for each desciptor (ctd, cst, fct and ftf) 
              a mean, standard deviation and the cubic root of the skew are 
              returned.
    '''
    
    #
    # ctd - centroid
    # cst - closest atom to centroid
    # fct - farthest atom to centroid
    # ftf - fartherst atom to fct 
    #

    distances_ctd = distances_to_point(coords, mean(coords, axis=0)) 

    distances_cst = distances_to_point(coords, coords[distances_ctd.argmin()])
    
    distances_fct = distances_to_point(coords, coords[distances_ctd.argmax()]) 
    
    distances_ftf = distances_to_point(coords, coords[distances_fct.argmax()]) 
    
    #
    # All this makes baby Jesus cry... it's 2x faster than usr_moments from 
    # Adrian by avoiding the list comprehension, but nobody will understand
    # what's going on here in 6 months...
    #
    arr = vstack((distances_ctd,distances_cst, distances_fct, distances_ftf))
    
    return hstack((arr.mean(axis=1), arr.std(axis=1), cbrt(skew(arr, axis=1))))[list((0,4,8,1,5,9,2,6,10,3,7,11))]
