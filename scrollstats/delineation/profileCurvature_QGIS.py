# For help:
# processing.algorithmHelp('grass7:r.param.scale')

import processing
import os

#winsize = range(3, 71, 2)
winsize = [29]

dem = '/Users/andrewvanderheiden/School/FLUD/BrazosScrolls/data/stratmap15_1m_merge_clip.tif'
out_folder = '/Users/andrewvanderheiden/School/FLUD/BrazosScrolls/data'


for i in winsize:
    temp_out_name = 'TEMP_{}px.tif'.format(i)

    # perform profc process
    profc = processing.run('grass7:r.param.scale',{
    'input': dem,
    'size': i,
    'method': 3,
    'output': os.path.join(out_folder, temp_out_name)
    })

    # Assign proj to raster - grass alg does not do this
    out_name = 'bend25_dem2015_profc_{}px.tif'.format(i)
    warp = processing.run('gdal:warpreproject',{
    'INPUT': profc['output'],
    'SOURCE_CRS':dem,
    'TARGET_CRS':dem,
    'OUTPUT': os.path.join(out_folder, out_name)
    })


    # Delete Temp files
    for temp in os.listdir(out_folder):
        if temp.split('.',1)[0] == temp_out_name.split('.')[0]:
            os.remove(os.path.join(out_folder, temp))
            print('Auto Deleted: {}'.format(temp))



    iface.addRasterLayer(warp['OUTPUT'], '{}px'.format(i))
