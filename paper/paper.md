---
title: 'ScrollStats: a python tool for quantifying scroll bar morphology on meandering rivers'
tags: 
  - python
  - scroll bars
  - river
  - meander
  - geomorphology
authors:
  - name: Andrew Vanderheiden
    orcid: 0009-0003-2466-8144 
    corresponding: true
    affiliation: '1'
  - name: Billy Hales
    orcid: 0000-0002-6591-9363 
    affiliation: '1'
  - name: Andrew Moodie
    orcid: 0000-0002-6745-036X
    affiliation: '1'
  - name: İnci Güneralp 
    orcid: 0000-0002-7268-2974
    affiliation: '1'
affiliations:
- name: Texas A&M University, United States
  index: '1'
date: 30 November 2024
bibliography: paper.bib

---
# Summary
Scroll bars are elongated arcuate topographic features deposited along the inner bank of meandering rivers. As the river continues to meander across the floodplain, the series of scroll bars deposited in its wake is known as ridge and swale topography. The ridge and swale topography, readily observed in LiDAR-derived digital elevation models (DEMs), contains a visually intuitive record of the river's migration history, and the specific morphology of each ridge may serve as a proxy for the hydrological, geomorphological, and sedimentological conditions under which each individual scroll bar is formed. The ridge crests, with their higher elevation relative to the swales also encourages the growth of colonizing vegetation, which in turn stabilizes the ridges and mitigate future erosion [@zen:2017]. While there has long been an interest in the formation and preservation of scroll bars, research into the specific drivers of ridge morphology and what information it may contain is new [@strick:2018; @nagy:2020]. 

# Statement of need
`ScrollStats` is an open-source python tool to quantify the morphology of scroll bars preserved in the ridge and swale topography commonly found in the floodplains of meandering rivers adjacent to the river channel. This quantification will allow researchers to investigate the relationships between ridge morphology and the factors affecting its formation such as the hydrology at the time of deposition, spatial variations in the river width, the channel curvature, the position along the meander bend, and the floodplain vegetation coverage and composition.

`ScrollStats` generates a series of migration pathways (an adaptation of the "erosion pathlines" from @hickin:1974) that trace the paths of migration across the bend from the channel centerline to the most ancestral ridge. These migration pathways are then used to sample the underlying DEM and binary ridge area raster to create a series of one-dimensional (1-D) signals of ridge elevation and ridge presence. Then, from each 1-D signal, the ridge's amplitude, width, and spacing (distance from the previous ridge) can be calculated at every intersection of the migration pathway and a ridge.

The intersections of migration pathways and ridge lines form a migrationally relevant grid, which allows for the measurements at each intersection to be aggregated to larger spatial scales (ridge-scale, transect-scale, bend-scale). This hierarchical spatial relationship enables researchers to study ridge morphology as it changes over time (from ridge to ridge) and along the channel (from migration pathway to migration pathway) and examine the associations between these changes in ridge morphology and the environmental factors affecting their formation. This allows for researchers to leverage the morphological information stored in the floodplains of meandering rivers to deduce past events such as changes in flow regimes, river planform and bend dynamics, sediment flux, and carbon storage and release. Such information has potential to also inform the predictions of future meander migration patterns and habitat suitability for riverine fauna and flora including riparian forests.

# Acknowledgements

# References 