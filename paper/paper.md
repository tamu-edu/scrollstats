---
title:
  "ScrollStats: a Python tool for quantifying scroll bar morphology on
  meandering rivers"
tags:
  - Python
  - scroll bars
  - river
  - meander
  - geomorphology
authors:
  - name: Andrew Vanderheiden
    orcid: 0009-0003-2466-8144
    corresponding: true
    affiliation: "1"
  - name: Billy Hales
    orcid: 0000-0002-6591-9363
    affiliation: "1"
  - name: Andrew J. Moodie
    orcid: 0000-0002-6745-036X
    affiliation: "1"
  - name: İnci Güneralp
    orcid: 0000-0002-7268-2974
    affiliation: "1"
affiliations:
  - name: Department of Geography, Texas A&M University, United States
    index: "1"
date: 14 February 2025
bibliography: paper.bib
---

# Summary

Scroll bars are elongated arcuate topographic features deposited along the inner
bank of meandering rivers. As the river continues to meander across the
floodplain, the series of scroll bars deposited in its wake is known as ridge
and swale topography. The ridge and swale topography, readily observed in
LiDAR-derived digital elevation models (DEMs), contains a visually intuitive
record of the river's migration history, and the specific morphology of each
ridge may serve as a proxy for the hydrological, geomorphological, and
sedimentological conditions under which each individual scroll bar is formed.
The ridge crests, with their higher elevation relative to the swales, also
encourage the growth of colonizing vegetation, which in turn stabilizes the
ridges and mitigate future erosion [@zen:2017]. While there has long been an
interest in the formation and preservation of scroll bars, research into the
specific drivers of ridge morphology and what information it may contain is new
[@strick:2018; @nagy:2020].

# Statement of need

`ScrollStats` is an open-source Python tool to quantify the morphology of scroll
bars preserved in the ridge and swale topography commonly found in the
floodplains of meandering rivers adjacent to the river channel. This
quantification will allow researchers to investigate the relationships between
ridge morphology and the environmental factors affecting its formation, such as
the hydrology at the time of deposition, spatial variations in the river width,
the channel curvature, the position along the meander bend, and the floodplain
vegetation coverage and composition.

`ScrollStats` generates a series of migration pathways (an adaptation of the
"erosion pathlines" from @hickin:1974) that trace the paths of migration across
the bend from the channel centerline to the most ancestral ridge
(\autoref{fig:figure1}). These migration pathways are then used to sample the
underlying DEM and binary ridge area raster to create a series of
one-dimensional (1-D) signals of ridge elevation and ridge presence
(\autoref{fig:figure2}). Then, from each 1-D signal, the ridge's amplitude,
width, and spacing (distance from the previous ridge) can be calculated at every
intersection of the migration pathway and a ridge (\autoref{fig:figure3}).

The intersections of migration pathways and ridge lines form a migrationally
relevant grid, which allows for the measurements at each intersection to be
aggregated to larger spatial scales (ridge-scale, transect-scale, bend-scale)
(\autoref{fig:figure4}). This hierarchical spatial relationship enables
researchers to study ridge morphology as it changes over time (from ridge to
ridge) and along the channel (from migration pathway to migration pathway) and
examine the associations between these changes in ridge morphology and the
environmental factors affecting their formation. This allows for researchers to
leverage the morphological information stored in the floodplains of meandering
rivers to deduce past events such as changes in flow regimes, river planform and
bend dynamics, sediment flux, and carbon storage and release. Such information
has potential to also inform the predictions of future meander migration
patterns and habitat suitability for riverine fauna and flora including riparian
forests.

# State of the field

To the authors' best knowledge, there do not exist any other software packages
purpose built to capture the variability in ridge morphometrics across scroll
bar floodplains. The methodology and analysis of [@strick:2018] was influential
in the creation of `ScrollStats`, but their analysis did not result in the
creation of a software package, so contribution was not possible.

`ScrollStats` is built upon the extensive scientific python ecosystem, and
specifically relies upon popular geospatial libraries (shapely, geopandas, and
rasterio) for spatial analysis and data manipulation. Users familiar with these
python libraries should find working with and extending `ScrollStats` to be an
intuitive experience.

# Software design

`ScrollStats` was built with interoperability and extensibility for the end-user
in mind. For example, the `delineation` subpackage, which is responsible for
delineating ridge areas from the input DEM, by default uses two classifier
functions to delineate ridge areas within the DEM: profile curvature and
residual topography. However, end users can extend ScrollStats by supplying
their own list of classier functions that have the same callable signature as
the default classifier functions:
`classifier_func(ElevationArray2D, **kwargs) -> BinaryArray2D`. Likewise, the
denoising process uses a default list of denoising functions that the user can
extend with functions using the following callable signature
`denoiser_func(BinaryArray2D, **kwargs) -> BinaryArray2D`

Similarly, the `transecting` subpackage generates migration pathways using the
vertical resultant calculations as described in [@hickin:1974]. However, if the
user would prefer to use a different method of calculating migration
trajectories from ridge to ridge, these alternative transects could be used as a
drop-in replacement to calculate ridge metrics instead - so long as their
vertices were coincident with the ridge lines they intersect.

The DataExtractor classes in the `ridge_metrics` subpackage were designed to
mirror the spatial scales at which they operate: Bend, Transect, and
Intersection. This design communicates 1) which code is responsible for
extracting information at the given scale and 2) what information is necessary
as input to make these calculations. This enables future developers to
easily identify where to focus efforts if they wish to extend the functionality
of ScrollStats or troubleshoot unexpected results.

# Research impact statement

`ScrollStats` has had limited impact on research community at the time of
publishing as it was not openly distributed beforehand. However, the core
methodological framework of `ScrollStats` as well as initial findings from its
use on meander bends from the Lower Brazos River, TX have been presented at the
Association of American Geographers Annual Meeting in 2021 and 2023 during its
development. Additionally, `ScrollStats` has been used in the completion of two
Masters theses in the FLUD Lab at Texas A&M University.

# Figures

![Ridgelines (dotted black) and channel centerline (solid grey) are manually digitized from interpretation of the DEM (Brazos River, Texas) and the binary ridge area raster (not pictured). `ScrollStats` then generates migration pathways (solid red) from equally spaced starting points along the centerline by "walking" up the floodplain from ridge to ridge (see Fig 3 from Hickin 1974 for transect generation procedure via calculation of vertical resultants). Ridge amplitude, width, and spacing are then calculated at each intersection (black dots) through analysis of the 1D signals generated by sampling both the DEM and binary ridge area raster along each transect. These calculations are shown for an example intersection (purple dot) of a ridge (solid blue) and migration pathway (thick solid red) in the following figures.\label{fig:figure1}](figs/digitized_dem.png)

![The 1D signals sampled from the DEM (solid black line) and binary ridge area raster (ridge areas shown in light grey patches) along the example migration pathway (solid red line) from Figure 1. The location of each ridge intersection along the migration pathway is shown with a black dot and dashed line. The zero point along the y axis starts at the intersection with the first ridge on the floodplain and increases with distance from the channel. Subjectively digitized ridge lines often, but do not always, fall within the bounds of the objective ridge area classification (see second to last grey patch near 400m). Ridge metrics are only calculated along the migration pathway for intersection points with the ridge lines. Ridge metric calculations are shown graphically for the example intersection (purple dot and blue line) on Figure 3.\label{fig:figure2}](figs/example_transect.png)

![Graphic representation of ridge metric calculations for the example intersection (purple dot). Amplitude (a; shown in yellow) is calculated by averaging the differences between the maximum elevation found within the corresponding ridge area (grey patch) and the minimum elevation values found in the preceding (a1) and following (a2) swale areas. Width (w; shown in purple) is the distance between the edges of the corresponding ridge area. Spacing (s; shown in green) is the distance between the intersection point and the adjacent intersection point closer to the channel.\label{fig:figure3}](figs/example_metric.png)

![Measures of ridge amplitude (orange), width (purple), and spacing (green) are shown at the intersection, ridge, and migration pathway scales. Aggregate values represent the medain value of each measurement taken at a ridge or migration pathway.\label{fig:figure4}](figs/example_output.png)

# AI usage disclosure

Generative AI was used to create a limited number of docstrings and provide
implementation examples of common packaging, distribution, and documentation
tools used in open source software. Generative AI was not used in the writing of
this manuscript or any other supporting materials.

# Acknowledgements

This work was supported by the T3 Triads for Transformation Program, Texas A&M
University. We would also like to acknowledge and thank the other members of the
FLUD Lab for their help and feedback through the process of conceptualization,
development, and testing for `ScrollStats`. Without their support, this tool
would not be what it is today.

# References
