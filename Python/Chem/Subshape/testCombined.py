import Chem
from Chem import AllChem
from Chem.PyMol import MolViewer
from Chem.Subshape import SubshapeBuilder,SubshapeObjects,SubshapeAligner
import cPickle,copy

m1 = Chem.MolFromMolFile('testData/square1.mol')
m2 = Chem.MolFromMolFile('testData/square2.mol')

b = SubshapeBuilder.SubshapeBuilder()
b.gridDims=(10.,10.,5)
b.gridSpacing=0.4
b.winRad=2.0
if 1:
  print 'm1:'
  s1 = b.GenerateSubshapeShape(m1)
  cPickle.dump(s1,file('testData/square1.shp.pkl','wb+'))
  print 'm2:'
  s2 = b.GenerateSubshapeShape(m2)
  cPickle.dump(s2,file('testData/square2.shp.pkl','wb+'))
  ns1 = b.CombineSubshapes(s1,s2)
  b.GenerateSubshapeSkeleton(ns1)
  cPickle.dump(ns1,file('testData/combined.shp.pkl','wb+'))
else:
  s1 = cPickle.load(file('testData/square1.shp.pkl','rb'))
  s2 = cPickle.load(file('testData/square2.shp.pkl','rb'))
  #ns1 = cPickle.load(file('testData/combined.shp.pkl','rb'))
  ns1=cPickle.load(file('testData/combined.shp.pkl','rb'))

v = MolViewer()
SubshapeObjects.DisplaySubshape(v,s1,'shape1')
SubshapeObjects.DisplaySubshape(v,ns1,'ns1')
#SubshapeObjects.DisplaySubshape(v,s2,'shape2')

a = SubshapeAligner.SubshapeAligner()
pruneStats={}
algs =a.GetSubshapeAlignments(None,ns1,m1,s1,b,pruneStats=pruneStats)
print len(algs)
print pruneStats

import os,tempfile,Geometry
fName = tempfile.mktemp('.grd')
Geometry.WriteGridToFile(ns1.coarseGrid.grid,fName)
v.server.loadSurface(fName,'coarse','',2.5)
os.unlink(fName)
fName = tempfile.mktemp('.grd')
Geometry.WriteGridToFile(ns1.medGrid.grid,fName)
v.server.loadSurface(fName,'med','',2.5)
os.unlink(fName)
