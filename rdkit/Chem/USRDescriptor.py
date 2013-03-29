from scipy.stats import skew
from scipy.special import cbrt
from numpy import mean, sqrt, vstack, concatenate, hstack, array
from rdkit import Chem
"""
Author Jan Domanski 
Inspired by 
http://hg.adrianschreyer.eu/usrcat/src/70e075d93cd25370e7ef93301d0e28d49a0851c2/usrcat/geometry.py?at=default
from Adrian Schreyer

# Feel free to remove the above, I just want to make it clear that I have inspired
# myself on Adrian's code and give him the credit.
"""


def _distances_to_point(coords, point):
    """
    Returns an array containing the distances of each coordinate in the input
    coordinates to the input point.
    """
    return sqrt(((coords-point)**2).sum(axis=1))


def USR(mol):
    """
    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
      Molecule object, with embedded at least one conformer
    
    Returns
    -------
    usr : np.array
      For each conformer emebedded in a molecule, returns 12-element descrptor 
    
    Calculate the ultrafast recognition for all the conformers of the molecule.
    You need to precompute the conformers, see for example:
      - http://www.rdkit.org/docs/Cookbook.html#parallel-conformation-generation
    
    Reference:
    J Comput Chem. 2007 Jul 30;28(10):1711-23.
    Ultrafast shape recognition to search compound databases for similar molecular shapes.
    Ballester PJ, Richards WG.  
    """
    assert isinstance(mol, Chem.rdchem.Mol)
    
    assert len(mol.GetAtoms()) >= 3, "Molecule has %d atoms, in order to calculate the USR at least 3 atoms are needed." % len(mol.GetAtoms()) 
    
    assert mol.GetConformers(), "Molecule needs to contain at least one 3D conformer"
    
    usr_list = list()
    for conf in mol.GetConformers():
        coords = array([(p.x, p.y, p.z) for p in [conf.GetAtomPosition(i) for i in xrange(conf.GetNumAtoms())]])
        usr_list.append(_usr_desciptor(coords))
    return array(usr_list)
  
def _usr_desciptor(coords):
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

    distances_ctd = _distances_to_point(coords, mean(coords, axis=0)) 

    distances_cst = _distances_to_point(coords, coords[distances_ctd.argmin()])
    
    distances_fct = _distances_to_point(coords, coords[distances_ctd.argmax()]) 
    
    distances_ftf = _distances_to_point(coords, coords[distances_fct.argmax()]) 
    
    #
    # All this makes baby Jesus cry... it's 2x faster than usr_moments from 
    # Adrian by avoiding the list comprehension, but nobody will understand
    # what's going on here in 6 months...
    #
    arr = vstack((distances_ctd,distances_cst, distances_fct, distances_ftf))
    
    return hstack((arr.mean(axis=1), arr.std(axis=1), cbrt(skew(arr, axis=1))))[list((0,4,8,1,5,9,2,6,10,3,7,11))]
