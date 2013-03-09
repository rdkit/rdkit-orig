
Installation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Below a number of installation recipies is presented, with varying degree of
complexity.

Ubuntu
******

Ubuntu 12.04
========================

Installation from the repositories
----------------------------------

RDKit is available via the Ubuntu repositories, to install::

    sudo apt-get install python-rdkit librdkit1 rdkit-data

Building from source
--------------------
    
If you want to build from source use the git/svn repos to get the code, or 
download a tar.gz file. First you want to install the prerequisites::

     sudo apt-get install flex bison build-essential python-numpy cmake \
                          python-dev sqlite3 libsqlite3-dev libboost-dev \
                          libboost-python-dev libboost-regex-dev

Fetch the source, here as tar.gz but you could use git clone::

    wget http://downloads.sourceforge.net/project/rdkit/rdkit/QX_20XX/RDKit_20XX_XX_X.tgz

Untar into /opt, or a different location of (like your home dir)::

    sudo tar xzvf RDKit_20XX_XX_X.tgz -C /opt
    
Set evironmental variables in ~/.bashrc ::

    export RDBASE=/opt/RDKit_20XX_XX_X
    export LD_LIBRARY_PATH=$RDBASE/lib:$LD_LIBRARY_PATH
    export PYTHONPATH=$RDBASE:$PYTHONPATH    

To build, compile and (optionally) test the code::

    cd $RDBASE
    mkdir build
    cd build
    cmake ..
    make # try `make -j 4` to use 4 processors for compilation
    make install
    ctest
                          
The custom build has been based on a `blogpost from the OPIG <http://blopig.com/blog/?p=315>`_.


Other Linux or Mac
******************

The instructions below are for the Q42009 release and subsequent releases.

Getting Ready
=============

 * Required packages:

   * cmake. You need version 2.6 (or more recent). http://www.cmake.org if your linux distribution doesn't have an appropriate package. 
     
     .. note:: It seems that v2.8 is a better bet than v2.6. It might be worth compiling your own copy of v2.8 even if v2.6 is already installed.
   
   * flex and bison. These are frequently already installed if you have the various pieces of the development environment installed. Note that some Redhat-based systems have an extremely ancient version of flex (v2.5.4, from 1997) installed; in order to build the RDKit on these systems you need to compile and install a more recent version. The source is available at http://flex.sourceforge.net.
   
   * The following are required if you are planning on using the Python wrappers
   
      * The python headers. This probably means that you need to install the python-dev package (or whatever it's called) for your linux distribution.
      * sqlite3. You also need the shared libraries. This may require that you install a sqlite3-dev package.
      * You need to have numpy (http://www.scipy.org/NumPy) installed. 
      
        .. note:: for building with XCode4 on the MacOS – there seems to be a problem with the version of numpy that comes with XCode4. Please see below in the (see :ref:`faq`) section for a workaround.
 * Optional packages
 
   * If you would like to install the RDKit InChI support (first available in the Q2 2011 release), follow the instructions in $RDBASE/External/INCHI-API to get a copy of the InChI source and put it in the appropriate place.

Installing Boost
================
If your linux distribution has a boost-devel package including the python and regex libraries, you can use that and save yourself the steps below. 


.. note:: if you *do* have a version of the boost libraries pre-installed and you want to use your own version, be careful when you build the code. We've seen at least one example on a Fedora system where cmake compiled using a user-installed version of boost and then linked against the system version. This led to segmentation faults. There is a workaround for this below in the (see :ref:`faq`) section.

  * download the boost source distribution from `the boost web site <http://www.boost.org>`_
  * extract the source somewhere on your machine (e.g. ``/usr/local/src/boost_1_45_0``)
  * build the required boost libraries:
  
    * ``cd $BOOST``
    * If you want to use the python wrappers: ``./bootstrap.sh --with-libraries=python,regex``
    * If not using the python wrappers: ``./bootstrap.sh --with-libraries=regex``
    * Building on 32 bit systems: ``./bjam install``
    * Building on 64 bit systems: ``./bjam address-model=64 cflags=-fPIC cxxflags=-fPIC install``

    If you have any problems with this step, check the boost `installation instructions <http://www.boost.org/more/getting_started/unix-variants.html>`_.

Building the Code
=================
  * follow the Installing Boost instructions above.
  * environment variables:
  
    * RDBASE: the root directory of the RDKit distribution (e.g. ~/RDKit)
    * *Linux:* LD_LIBRARY_PATH: make sure it includes $RDBASE/lib and wherever the boost shared libraries were installed
    * *Mac:* DYLD_LIBRARY_PATH: make sure it includes $RDBASE/lib and wherever the boost shared libraries were installed
    * The following are required if you are planning on using the Python wrappers:
      * PYTHONPATH: make sure it includes $RDBASE
  * Building:
  
    * cd to $RDBASE
    * ``mkdir build``
    * ``cd build``
    * ``cmake ..`` : See the section below on configuring the build if you need to specify a non-default version of python or if you have boost in a non-standard location
    * ``make`` : this builds all libraries, regression tests, and wrappers (by default).
    * ``make install``

See below for a list of [#Frequently_Encountered_Problems frequently encountered problems] and solutions.

Testing the Build (optional, but recommended)
=============================================
  * cd to $RDBASE/build and do ``ctest``
  * you're done!

Advanced
========

Specifying an alternate Boost installation
------------------------------------------

You need to tell cmake where to find the boost libraries and header files:

If you have put boost in /opt/local, the cmake invocation would look like::

    cmake -DBOOST_ROOT=/opt/local ..

Specifying an alternate Python installation
-------------------------------------------

You need to tell cmake where to find the python library it should link against and the python header files.

Here's a sample command line::

    cmake -D PYTHON_LIBRARY=/usr/lib/python2.5/config/libpython2.5.a -D PYTHON_INCLUDE_DIR=/usr/include/python2.5/ -D PYTHON_EXECUTABLE=/usr/bin/python ..

The ``PYTHON_EXECUTABLE`` part is optional if the correct python is the first version in your PATH.

Disabling the Python wrappers
-----------------------------

You can completely disable building of the python wrappers by setting the configuration variable RDK_BUILD_PYTHON_WRAPPERS to nil::

    cmake -D RDK_BUILD_PYTHON_WRAPPERS= ..

Building the Java wrappers
--------------------------

*Additional Requirements*


* SWIG v2.0.x: http://www.swig.org
* Junit: get a copy of the junit .jar file from https://github.com/KentBeck/junit/downloads and put it in the directory ``$RDBASE/External/java_lib`` (you will need to create the directory) and rename it to junit.jar.

*Building*

* When you invoke cmake add ``-D RDK_BUILD_SWIG_WRAPPERS=ON`` to the arguments. 
  For example::
    cmake -D RDK_BUILD_SWIG_WRAPPERS=ON ..
* Build and install normally using `make`. The directory ``$RDBASE/Code/JavaWrappers/gmwrapper`` will contain the three required files: libGraphMolWrap.so (libGraphMolWrap.jnilib on the Mac), org.RDKit.jar, and org.RDKitDoc.jar.

*Using the wrappers*

To use the wrappers, the three files need to be in the same directory, and that should be on your CLASSPATH and in the java.library.path. An example using jython::

    % CLASSPATH=$CLASSPATH:$RDBASE/Code/JavaWrappers/gmwrapper/org.RDKit.jar; jython -Djava.library.path=$RDBASE/Code/JavaWrappers/gmwrapper
    Jython 2.2.1 on java1.6.0_20
    Type "copyright", "credits" or "license" for more information.
    >>> from org.RDKit import *
    >>> from java import lang
    >>> lang.System.loadLibrary('GraphMolWrap')
    >>> m = RWMol.MolFromSmiles('c1ccccc1')
    >>> m.getNumAtoms()
    6L



.. _faq:

Frequently Encountered Problems
===============================


In each case I've replaced specific pieces of the path with ``...``.

*Problem:* ::

    Linking CXX shared library libSLNParse.so
    /usr/bin/ld: .../libboost_regex.a(cpp_regex_traits.o): relocation R_X86_64_32S against `std::basic_string<char, std::char_traits<char>, std::allocator<char> >::_Rep::_S_empty_rep_storage' can not be used when making a shared object; recompile with -fPIC
    .../libboost_regex.a: could not read symbols: Bad value
    collect2: ld returned 1 exit status
    make[2]: *** [Code/GraphMol/SLNParse/libSLNParse.so] Error 1
    make[1]: *** [Code/GraphMol/SLNParse/CMakeFiles/SLNParse.dir/all] Error 2
    make: *** [all] Error 2


*Solution:*

Add this to the arguments when you call cmake: ``-DBoost_USE_STATIC_LIBS=OFF``

`more information here <http://www.mail-archive.com/rdkit-discuss@lists.sourceforge.net/msg01119.html>`_

----

*Problem:* ::


     .../Code/GraphMol/Wrap/EditableMol.cpp:114:   instantiated from here
     .../boost/type_traits/detail/cv_traits_impl.hpp:37: internal compiler error: in make_rtl_for_nonlocal_decl, at cp/decl.c:5067
    Please submit a full bug report,
    with preprocessed source if appropriate.
    See <URL:http://bugzilla.redhat.com/bugzilla> for instructions.
    Preprocessed source stored into /tmp/ccgSaXge.out file, please attach this to your bugreport.
    make[2]: *** [Code/GraphMol/Wrap/CMakeFiles/rdchem.dir/EditableMol.cpp.o] Error 1
    make[1]: *** [Code/GraphMol/Wrap/CMakeFiles/rdchem.dir/all] Error 2
    make: *** [all] Error 2


*Solution:*

Add ``#define BOOST_PYTHON_NO_PY_SIGNATURES`` at the top of ``Code/GraphMol/Wrap/EditableMol.cpp``

`more information here <http://www.mail-archive.com/rdkit-discuss@lists.sourceforge.net/msg01178.html>`_


----

*Problem:*

Your system has a version of boost installed in /usr/lib, but you would like to force the RDKit to use a more recent one.

*Solution:*

This can be solved by using cmake version 2.8.3 (or more recent) and providing the ``-D Boost_NO_SYSTEM_PATHS=ON`` argument::

    cmake -D BOOST_ROOT=/usr/local -D Boost_NO_SYSTEM_PATHS=ON ..


----

*Problem:*

Building on the Mac with XCode 4

The problem seems to be caused by the version of numpy that is distributed with XCode 4, so you need to build a fresh copy.


*Solution:*
Get a copy of numpy and build it like this as root:
as root::

    export MACOSX_DEPLOYMENT_TARGET=10.6
    export LDFLAGS="-Wall -undefined dynamic_lookup -bundle -arch x86_64"
    export CFLAGS="-arch x86_64"
    ln -s /usr/bin/gcc /usr/bin/gcc-4.2
    ln -s /usr/bin/g++ /usr/bin/g++-4.2
    python setup.py build
    python setup.py install


Be sure that the new numpy is used in the build::

    PYTHON_NUMPY_INCLUDE_PATH        /Library/Python/2.6/site-packages/numpy/core/include

and is at the beginning of the PYTHONPATH::

    export PYTHONPATH="/Library/Python/2.6/site-packages:$PYTHONPATH"

Now it's safe to build boost and the RDKit.
