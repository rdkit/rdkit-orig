# $Id$
#
#  Copyright (C) 2003-2006  Rational Discovery LLC
#
#   @@ All Rights Reserved @@
#  This file is part of the RDKit.
#  The contents are covered by the terms of the BSD license
#  which is included in the file license.txt, found at the root
#  of the RDKit source tree.
#
""" unit testing code for USR descriptors parameter calculation
"""
import unittest, numpy 
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors

class TestCase(unittest.TestCase):
  def setUp(self):
    pass
  def test_snity(self):
    # Cannot compute USR for molecule with less than 3 atoms
    m = Chem.MolFromSmiles("C")
    self.assertRaises(AssertionError, Descriptors.USR, m)
    
    # Cannot compute USR for molecule without conformers (1)
    m = Chem.MolFromSmiles("CCC")
    self.assertRaises(AssertionError, Descriptors.USR, m)

    # Cannot compute USR for molecule without conformers (2)
    m = Chem.AddHs(m)
    self.assertRaises(AssertionError, Descriptors.USR, m)
        
    # Can compute for 2D molecule
    # FIXME this may lead to trouble with users using 2D molecules...
    AllChem.Compute2DCoords(m)
    usr = Descriptors.USR(m)
    self.assertEqual(usr.shape, (1,12), "The shape of USR descriptor for CCC should be (1,12) found %s" % str(usr.shape))
  def test_numerics(self):
    m = Chem.MolFromSmiles("C1CCCCC1")
    m = Chem.AddHs(m)
    AllChem.Compute2DCoords(m)
    usr = Descriptors.USR(m)

    #
    # Comparing to results produced by Adrian Schreyer's code
    # http://hg.adrianschreyer.eu/usrcat/src/70e075d93cd25370e7ef93301d0e28d49a0851c2/usrcat/geometry.py?at=default
    #     
    desired_usr = numpy.array([ 2.37938524,  0.62181927, -0.89089872,  2.63773456,  1.1577952 , \
       -0.6937349 ,  3.38248245,  1.59816952, -0.72933115,  3.38248245, \
        1.59816952, -0.72933115])

    numpy.testing.assert_allclose(usr[0], desired_usr, rtol=1e-8) # rtol is relative tolerance, this parameter will be sensitive to machine precision 

    numConfs=10
    AllChem.EmbedMultipleConfs(m, numConfs=numConfs)
    usr = Descriptors.USR(m)
    self.assertEqual(usr.shape, (numConfs,12), "For molecule with multiple (%d) conformers embedded, the shape of the descriptor array should be %s, is %s" % (numConfs, str((numConfs,12)), str(usr.shape)))  


if __name__ == '__main__':
  unittest.main()
  
