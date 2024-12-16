PUBLIC WIFI USER GUIDE
======================

A warning to the user: this is Not A Good README


Table of Contents
-----------------
1. Overview
1. Installation
1. Usage
1. Components
1. TODO



Overview
--------

PUBLIC WIFI is a tool for performing self-referencing PSF subtraction on a
survey of many targets (typically a star, brown dwarf, or rogue planet), all of
whom may or may not have companions. Each target may be imaged once, or more
than once. PUBLIC WIFI takes as input a catalog that has one row for each time
an target is detected. Each row must contain at least the name of the target
(which will be the same on each row), the name of the file containing the
relevant exposure for that detection, and the x/y position of the target's
detection in the exposure. Other columns might include something to use for
matching references (e.g. filter or instrument). It is the user's
responsibility to make this catalog.

When the catalog is provided to PUBLIC WIFI, it creates a Star object for each
target that contains the corresponding catalog rows in the `star.cat` attribute.
Each star queries the other stars' attributes during catalog processing to
assemble suitable reference PSFs for PSF subtraction and point source detection.
Once a Star object is given the appropriate reference information, it has all
the methods required to perform PSF subtraction on itself.

Installation
------------

A conda environment that is compatible with this version of PUBLIC WIFI
is provided in ``docs/conda-public_wifi.yml`` (it has more packages than
are strictly necessary). You can create the environment from a terminal
with the command ``conda env create -f conda-public_wifi.yml``. You can
then activate the environment with ``conda activate public_wifi``.

Some packages might need to be installed manually by the user.

USAGE
-----

The user must provide PUBLIC WIFI with an input point source catalog in the form of a pandas DataFrame. This is a catalog of all the point source detections relevant to the self-referencing survey. Targets may have more than one detection. Each row corresponds to one detection, so a target that is rpresent in multiple exposures will have more than one row.


Components
----------


TODO
----

   1. Right now, the user can only specify which catalog columns to use for
      matching references (e.g. match on the same filter). The user should be able
      to write their own function to match references that operates on the catalog
      and returns True or False.

