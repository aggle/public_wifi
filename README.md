PUBLIC WIFI USER GUIDE
======================

A warning to the user: This is very much a work in progress. I will do my best
to keep this README current, but it may be outdated.


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
matching references (e.g. filter or instrument). It is the user's responsibility
to make this catalog.

When the catalog is provided to PUBLIC WIFI, it creates a Star object for each
target that contains the corresponding catalog rows in the `star.cat` attribute.
Each star queries the other stars' attributes during catalog processing to
assemble suitable reference PSFs for PSF subtraction and point source detection.
Once a Star object is given the appropriate reference information, it has all
the methods required to perform PSF subtraction on itself.

When using the graphical data exploration interface, the dashboard takes a star
object and displays a selection of its attributes.

Installation
------------

To install `public_wifi`, clone this repository to a directory of your choice.
Navigate to that directory in a terminal, activate your preferred python
environment, and run the command `pip install .`

BASIC USAGE
-----------

The user must provide PUBLIC WIFI with an input point source catalog in the form
of a pandas DataFrame. This is a catalog of all the point source detections
relevant to the self-referencing survey. Targets may have more than one
detection. Each row corresponds to one detection, so a target that is rpresent
in multiple exposures will have more than one row. This catalog is provided to
the `catalog_procesisng::process_catalog()` method, which will return a pandas
Series with a `starclass::Star` object in each entry.

The interactive visualizer can be started with the script provided at
`public_wifi/src/star_dashboard_script.py`. Users will want to make a copy of
this script for their own purposes, and write their own implementation of
`catproc.load_catalog()`. To launch the dashboard, run this script from a
terminal in the appropriate python environment. `bokeh` is required, but its
configuration is beyond the scope of this README.
 
### Input catalog ###

As stated above, the main work of the user is creating the initial catalog. The
catalog must have a column for an identifier for the targets, a column with the
name of the file containing the exposure, and columns for the x and y position
of the target in that exposure. All other data are optional.

### Process the catalog ###

One function will initialize the `Star` objects, organize PSF subtraction, and
run the detection routines: `catalog_processing.process_catalog()`. Provide it
with the threshold on the similarity score, the minimum number of references to
use regardless of score, an SNR detection threshold, and a "# modes" detection
threshold. This method is essentially a wrapper around the three major analysis
functions:
- `catalog_initialization()`: read the catalog, create a `Star` object for each
  target, and prepare the data for PSF subtraction
- `catalog_subtraction()` : run the PSF subtraction algorithm on the exposures
  for each Star
- `catalog_detection()` : run the candidate detection algorithm on the PSF
  subtraction data products

The `match_references_on` argument is used for listing catalog columns that
should have the same value for target and reference - for example, "filter".

This function returns a pd.Series where each entry is a Star object. The
analysis results are saved in the `star.results` dataframe. The number of rows
in this dataframe corresponds to the number of different matching categories
used (for example, if two filters were analyzed separately, there will be two
rows).

TODO
----
- Move all injection and recovery code out of the Star class
