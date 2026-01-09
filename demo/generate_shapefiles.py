import numpy as np


def plot_glacier_study():
    import pygmt
    import geopandas as gpd

    fig = pygmt.Figure()
    pygmt.config(MAP_HEADING_OFFSET=0, MAP_TITLE_OFFSET=-0.2)
    pygmt.config(FONT_ANNOT='10p', COLOR_NAN='white')
    pygmt.makecpt(cmap='wysiwyg', series=[-5, 7], background='o')

    region = [-80, -5, 55, 85]
    pj = 'X7c/8c'
    fig.basemap(region=region, projection=pj,
                frame=['WSne', 'xa10f5+lLongitude (\\260 E)', 'ya5f5+lLatitude (\\260 N)'])

    fig.coast(shorelines="1/0.2p", region=region, projection=pj, water="skyblue")

    gdf = gpd.read_file('/media/user/Backup Plus/glacier_shp/Greenland/greenland.shp').to_crs(crs='epsg:4326')
    gdf.replace('CE', 0.0, inplace=True)
    gdf.replace('CW', 1.0, inplace=True)
    gdf.replace('NE', 2.0, inplace=True)
    gdf.replace('NO', 3.0, inplace=True)
    gdf.replace('NW', 4.0, inplace=True)
    gdf.replace('SE', 5.0, inplace=True)
    gdf.replace('SW', 6.0, inplace=True)

    fig.plot(data=gdf, pen="0.2p,black", fill='+z', cmap=True, aspatial='Z=SUBREGION1', projection=pj, close=True)

    # fig.plot(data=gdf.boundary, pen="0.05p,black", projection=pj)

    gdf = gpd.read_file('/media/user/Backup Plus/glacier_shp/Greenland/greenland.shp').to_crs(crs='epsg:4326')
    for i in range(7):
        nn = gdf.SUBREGION1[i]
        xy = gdf[gdf.SUBREGION1 == nn].centroid
        fig.text(x=xy.x, y=xy.y, text="%s" % nn, font='8p,black')

    fig.show()

    pass


def plot_us():
    import pygmt
    import geopandas as gpd
    import pandas as pd

    fig = pygmt.Figure()
    pygmt.config(MAP_HEADING_OFFSET=0, MAP_TITLE_OFFSET=-0.2)
    pygmt.config(FONT_ANNOT='10p', COLOR_NAN='white')
    pygmt.makecpt(cmap='wysiwyg', series=[0, 56], background='o')

    region = [-130, -65, 20, 55]
    # region = 'g'
    pj = 'Q12c'
    fig.basemap(region=region, projection=pj,
                frame=['WSne', 'xa10f5+lLongitude (\\260 E)', 'ya5f5+lLatitude (\\260 N)'])

    fig.coast(shorelines="1/0.2p", region=region, projection=pj, water="skyblue")

    gdf = gpd.read_file('/media/user/Backup Plus/GRACE/shapefiles/USgrid').to_crs(crs='epsg:4326')
    # gdf['OBJECTID'] = gdf['OBJECTID'].astype(float)
    # gdf['OB']
    # gdf.replace('CE', 0.0, inplace=True)
    # gdf.replace('CW', 1.0, inplace=True)
    # gdf.replace('NE', 2.0, inplace=True)
    # gdf.replace('NO', 3.0, inplace=True)
    # gdf.replace('NW', 4.0, inplace=True)
    # gdf.replace('SE', 5.0, inplace=True)
    # gdf.replace('SW', 6.0, inplace=True)

    # fig.plot(data=gdf, pen="0.2p,black", fill='+z', cmap=True, aspatial='Z=OBJECTID', projection=pj, close=True)

    # fig.plot(data=gpd.GeoSeries(gdf.unary_union.boundary), pen="0.05p,black", projection=pj)

    fig.plot(data=gdf.boundary, pen="0.5p,red", projection=pj)

    gdf = gpd.read_file('/media/user/Backup Plus/GRACE/shapefiles/USgrid').to_crs(crs='epsg:4326')
    for i in range(gdf.shape[0]):
        nn = gdf.ID[i]
        xy = gdf[gdf.ID == nn].centroid
        fig.text(x=xy.x, y=xy.y, text="%s" % nn, font='5p,black')

    fig.show()

    pass


def plot_Africa():
    import pygmt
    import geopandas as gpd
    import pandas as pd

    fig = pygmt.Figure()
    pygmt.config(MAP_HEADING_OFFSET=0, MAP_TITLE_OFFSET=-0.2)
    pygmt.config(FONT_ANNOT='10p', COLOR_NAN='white')
    pygmt.makecpt(cmap='wysiwyg', series=[0, 56], background='o')

    region = [-20, 60, -40, 40]
    # region = 'g'
    pj = 'Q12c'
    fig.basemap(region=region, projection=pj,
                frame=['WSne', 'xa10f5+lLongitude (\\260 E)', 'ya5f5+lLatitude (\\260 N)'])

    fig.coast(shorelines="1/0.2p", region=region, projection=pj, water="skyblue")

    # gdf = gpd.read_file('/media/user/Backup Plus/GRACE/shapefiles/continent-poly.zip').to_crs(crs='epsg:4326')

    # gdf=gdf[gdf.CONTINENT=='Africa']
    # gdf.to_file('/media/user/Backup Plus/GRACE/shapefiles/continent_Africa')
    gdf = gpd.read_file('/media/user/Backup Plus/GRACE/shapefiles/Africa5').to_crs(crs='epsg:4326')
    # gdf['OBJECTID'] = gdf['OBJECTID'].astype(float)
    # gdf['OB']
    # gdf.replace('CE', 0.0, inplace=True)
    # gdf.replace('CW', 1.0, inplace=True)
    # gdf.replace('NE', 2.0, inplace=True)
    # gdf.replace('NO', 3.0, inplace=True)
    # gdf.replace('NW', 4.0, inplace=True)
    # gdf.replace('SE', 5.0, inplace=True)
    # gdf.replace('SW', 6.0, inplace=True)

    # fig.plot(data=gdf, pen="0.2p,black", fill='+z', cmap=True, aspatial='Z=OBJECTID', projection=pj, close=True)

    # bb = max(gdf.unary_union.geoms, key=lambda a: a.area)

    # bb = gdf.unary_union

    # fig.plot(data=gpd.GeoSeries(bb), pen="0.5p,red", projection=pj)

    fig.plot(data=gdf.boundary, pen="0.5p,red", projection=pj)

    gdf = gpd.read_file('/media/user/Backup Plus/GRACE/shapefiles/Africa5').to_crs(crs='epsg:4326')
    for i in range(gdf.shape[0]):
        nn = gdf.ID[i]
        xy = gdf[gdf.ID == nn].centroid
        fig.text(x=xy.x, y=xy.y, text="%s" % nn, font='5p,black')

    fig.show()

    pass


def USgrid():
    from src_auxiliary.create_shp import basin2grid_shp

    b2s = basin2grid_shp().configure(new_basin_name='UnitedStatesGrid',
                                     original_shp='/media/user/Backup Plus/GRACE/shapefiles/tl_2012_us_state.zip')
    b2s.set_modify_func(func=basin2grid_shp.example_func1)
    b2s.create_shp()

    pass


def Africagrid():
    from src_auxiliary.create_shp import basin2grid_shp

    b2s = basin2grid_shp(grid=(5, 5)).configure(new_basin_name='Africa_subbasins',
                                                original_shp='/media/user/Backup Plus/GRACE/shapefiles/continent_Africa.zip')
    b2s.set_modify_func(func=basin2grid_shp.example_func1)
    b2s.create_shp()

    pass


def Brahmaputra_grid(res=5):
    from src_auxiliary.create_shp import basin2grid_shp
    # res=5
    b2s = basin2grid_shp(grid=(res, res)).configure(new_basin_name='Brahmaputra_%s' % res,
                                                    original_shp='/home/user/codes/py-w3ra/data/basin/shp/Brahmaputra_3_shapefiles/Brahmaputra_3_subbasins.shp')
    b2s.set_modify_func(func=basin2grid_shp.example_func1)
    b2s.create_shp(out_dir='/media/user/My Book/Fan/ESA_SING/shapefiles/Brahmaputra/Grid_%s' % res)

    pass


def Brahmaputra_subbasins_show():
    import pygmt
    import geopandas as gpd
    import pandas as pd

    fig = pygmt.Figure()
    pygmt.config(MAP_HEADING_OFFSET=0, MAP_TITLE_OFFSET=-0.2)
    pygmt.config(FONT_ANNOT='10p', COLOR_NAN='white')
    pygmt.makecpt(cmap='wysiwyg', series=[0, 56], background='o')

    region = [87, 98, 21, 32]
    # region = 'g'
    pj = 'Q12c'
    fig.basemap(region=region, projection=pj,
                frame=['WSne', 'xa5f5+lLongitude (\\260 E)', 'ya5f5+lLatitude (\\260 N)'])

    fig.coast(shorelines="1/0.2p", region=region, projection=pj, water="skyblue")

    gdf = gpd.read_file('/home/user/codes/py-w3ra/data/basin/shp/Brahmaputra_3_shapefiles/Brahmaputra_3_subbasins.shp')
    # gdf['OBJECTID'] = gdf['OBJECTID'].astype(float)
    # gdf['OB']
    # gdf.replace('CE', 0.0, inplace=True)
    # gdf.replace('CW', 1.0, inplace=True)
    # gdf.replace('NE', 2.0, inplace=True)
    # gdf.replace('NO', 3.0, inplace=True)
    # gdf.replace('NW', 4.0, inplace=True)
    # gdf.replace('SE', 5.0, inplace=True)
    # gdf.replace('SW', 6.0, inplace=True)

    # fig.plot(data=gdf, pen="0.2p,black", fill='+z', cmap=True, aspatial='Z=OBJECTID', projection=pj, close=True)

    # fig.plot(data=gpd.GeoSeries(gdf.unary_union.boundary), pen="0.05p,black", projection=pj)

    fig.plot(data=gdf.boundary, pen="2p,red", projection=pj)

    gdf = gpd.read_file('/home/user/codes/py-w3ra/data/basin/shp/Brahmaputra_3_shapefiles/Brahmaputra_3_subbasins.shp')
    for i in range(gdf.shape[0]):
        nn = gdf.ID[i]
        xy = gdf[gdf.ID == nn].centroid
        fig.text(x=xy.x, y=xy.y, text="%s" % nn, font='15p,black')

    fig.savefig('/media/user/My Book/Fan/ESA_SING/shapefiles/Brahmaputra/subbasins_3.png')
    fig.show()

    pass


def Brahmaputra_grid_show(res=5):
    import pygmt
    import geopandas as gpd
    import pandas as pd

    fig = pygmt.Figure()
    pygmt.config(MAP_HEADING_OFFSET=0, MAP_TITLE_OFFSET=-0.2)
    pygmt.config(FONT_ANNOT='10p', COLOR_NAN='white')
    pygmt.makecpt(cmap='wysiwyg', series=[0, 56], background='o')

    # res=5

    region = [87, 98, 21, 32]
    # region = 'g'
    pj = 'Q12c'
    fig.basemap(region=region, projection=pj,
                frame=['WSne', 'xa5f5+lLongitude (\\260 E)', 'ya5f5+lLatitude (\\260 N)'])

    fig.coast(shorelines="1/0.2p", region=region, projection=pj, water="skyblue")

    gdf = gpd.read_file(
        '/media/user/My Book/Fan/ESA_SING/shapefiles/Brahmaputra/Grid_%s/Brahmaputra_%s.shp' % (res, res))
    # gdf['OBJECTID'] = gdf['OBJECTID'].astype(float)
    # gdf['OB']
    # gdf.replace('CE', 0.0, inplace=True)
    # gdf.replace('CW', 1.0, inplace=True)
    # gdf.replace('NE', 2.0, inplace=True)
    # gdf.replace('NO', 3.0, inplace=True)
    # gdf.replace('NW', 4.0, inplace=True)
    # gdf.replace('SE', 5.0, inplace=True)
    # gdf.replace('SW', 6.0, inplace=True)

    # fig.plot(data=gdf, pen="0.2p,black", fill='+z', cmap=True, aspatial='Z=OBJECTID', projection=pj, close=True)

    # fig.plot(data=gpd.GeoSeries(gdf.unary_union.boundary), pen="0.05p,black", projection=pj)

    fig.plot(data=gdf.boundary, pen="2p,red", projection=pj)

    gdf = gpd.read_file(
        '/media/user/My Book/Fan/ESA_SING/shapefiles/Brahmaputra/Grid_%s/Brahmaputra_%s.shp' % (res, res))
    for i in range(gdf.shape[0]):
        nn = gdf.ID[i]
        xy = gdf[gdf.ID == nn].centroid
        fig.text(x=xy.x, y=xy.y, text="%s" % nn, font='15p,black')

    fig.savefig('/media/user/My Book/Fan/ESA_SING/shapefiles/Brahmaputra/Grid_%s.png' % res)
    fig.show()

    pass


def Danube_grid(res=5):
    from src_auxiliary.create_shp import basin2grid_shp
    # res=5
    b2s = basin2grid_shp(grid=(res, res)).configure(new_basin_name='Danube%sdegree_subbasins' % res,
                                                    original_shp='/home/user/codes/py-w3ra/data/basin/shp/DRB_3_shapefiles/DRB_subbasins.shp')

    # b2s.set_modify_func(func=basin2grid_shp.example_func1)
    # b2s.set_modify_func(func=None)
    # b2s.create_shp(out_dir='/media/user/My Book/Fan/ESA_SING/shapefiles/Danube/Grid_%s' % res)

    def func(gdf):
        import numpy as np
        gdf['ID'] = np.array([1, 1, 2, 3, 4, 4, 5, 5, 6, 7, 7])
        md = gdf.dissolve(by='ID').reset_index()
        return md

    b2s.set_modify_func(func=func)
    b2s.create_shp(out_dir='/media/user/My Book/Fan/ESA_SING/shapefiles/Danube/Grid_%s' % res)

    pass


def Danube_subbasins_show():
    import pygmt
    import geopandas as gpd
    import pandas as pd

    fig = pygmt.Figure()
    pygmt.config(MAP_HEADING_OFFSET=0, MAP_TITLE_OFFSET=-0.2)
    pygmt.config(FONT_ANNOT='10p', COLOR_NAN='white')
    pygmt.makecpt(cmap='wysiwyg', series=[0, 56], background='o')

    region = [6, 30, 40, 54]
    # region = 'g'
    pj = 'Q12c'
    fig.basemap(region=region, projection=pj,
                frame=['WSne', 'xa5f5+lLongitude (\\260 E)', 'ya5f5+lLatitude (\\260 N)'])

    fig.coast(shorelines="1/0.2p", region=region, projection=pj, water="skyblue")

    gdf = gpd.read_file('/home/user/codes/py-w3ra/data/basin/shp/DRB_3_shapefiles/DRB_subbasins.shp')
    # gdf['OBJECTID'] = gdf['OBJECTID'].astype(float)
    # gdf['OB']
    # gdf.replace('CE', 0.0, inplace=True)
    # gdf.replace('CW', 1.0, inplace=True)
    # gdf.replace('NE', 2.0, inplace=True)
    # gdf.replace('NO', 3.0, inplace=True)
    # gdf.replace('NW', 4.0, inplace=True)
    # gdf.replace('SE', 5.0, inplace=True)
    # gdf.replace('SW', 6.0, inplace=True)

    # fig.plot(data=gdf, pen="0.2p,black", fill='+z', cmap=True, aspatial='Z=OBJECTID', projection=pj, close=True)

    # fig.plot(data=gpd.GeoSeries(gdf.unary_union.boundary), pen="0.05p,black", projection=pj)

    fig.plot(data=gdf.boundary, pen="2p,red", projection=pj)

    gdf = gpd.read_file('/home/user/codes/py-w3ra/data/basin/shp/DRB_3_shapefiles/DRB_subbasins.shp')
    for i in range(gdf.shape[0]):
        nn = gdf.ID[i]
        xy = gdf[gdf.ID == nn].centroid
        fig.text(x=xy.x, y=xy.y, text="%s" % nn, font='15p,black')

    fig.savefig('/media/user/My Book/Fan/ESA_SING/shapefiles/Danube/subbasins_3.png')
    fig.show()

    pass


def Danube_grid_show(res=5):
    import pygmt
    import geopandas as gpd
    import pandas as pd

    fig = pygmt.Figure()
    pygmt.config(MAP_HEADING_OFFSET=0, MAP_TITLE_OFFSET=-0.2)
    pygmt.config(FONT_ANNOT='10p', COLOR_NAN='white')
    pygmt.makecpt(cmap='wysiwyg', series=[0, 56], background='o')

    # res=5

    region = [6, 30, 40, 54]
    # region = 'g'
    pj = 'Q12c'
    fig.basemap(region=region, projection=pj,
                frame=['WSne', 'xa5f5+lLongitude (\\260 E)', 'ya5f5+lLatitude (\\260 N)'])

    fig.coast(shorelines="1/0.2p", region=region, projection=pj, water="skyblue")

    gdf = gpd.read_file(
        '/media/user/My Book/Fan/ESA_SING/shapefiles/Danube/Grid_%s/Danube%sdegree_subbasins.shp' % (res, res))
    # gdf['OBJECTID'] = gdf['OBJECTID'].astype(float)
    # gdf['OB']
    # gdf.replace('CE', 0.0, inplace=True)
    # gdf.replace('CW', 1.0, inplace=True)
    # gdf.replace('NE', 2.0, inplace=True)
    # gdf.replace('NO', 3.0, inplace=True)
    # gdf.replace('NW', 4.0, inplace=True)
    # gdf.replace('SE', 5.0, inplace=True)
    # gdf.replace('SW', 6.0, inplace=True)

    # fig.plot(data=gdf, pen="0.2p,black", fill='+z', cmap=True, aspatial='Z=OBJECTID', projection=pj, close=True)

    # fig.plot(data=gpd.GeoSeries(gdf.unary_union.boundary), pen="0.05p,black", projection=pj)

    fig.plot(data=gdf.boundary, pen="2p,red", projection=pj)

    gdf = gpd.read_file(
        '/media/user/My Book/Fan/ESA_SING/shapefiles/Danube/Grid_%s/Danube%sdegree_subbasins.shp' % (res, res))
    for i in range(gdf.shape[0]):
        nn = gdf.ID[i]
        xy = gdf[gdf.ID == nn].centroid
        fig.text(x=xy.x, y=xy.y, text="%s" % nn, font='7p,black')

    fig.savefig('/media/user/My Book/Fan/ESA_SING/shapefiles/Danube/Grid_%s.png' % res)
    fig.show()

    pass


def box2shp(box_area=[70.1, 33.9, -11.1, 45.1]):
    """
    Generate a shp file from a box boundary.

    For example: box = [76.1, 33.9, -11.1, 45.1] ==> [up (lat), down (lat), left (lon), right (lon)]
    """

    import numpy as np
    import geopandas as gpd
    from shapely import box

    poly = box(xmin=box_area[2], ymin=box_area[0], xmax=box_area[3], ymax=box_area[1])
    d = {'ID': [0], 'geometry': [poly]}
    gdf = gpd.GeoDataFrame(d, crs='epsg:4326')

    # '''visualization'''
    # import pygmt
    # import geopandas as gpd
    # fig = pygmt.Figure()
    # pygmt.config(MAP_HEADING_OFFSET=0, MAP_TITLE_OFFSET=-0.2)
    # pygmt.config(FONT_ANNOT='10p', COLOR_NAN='white')
    # pygmt.makecpt(cmap='wysiwyg', series=[0, 56], background='o')
    # region = 'g'
    # pj = "Q30/-20/12c"
    # fig.basemap(region=region, projection=pj,
    #             frame=['WSne', 'xa10f5+lLongitude (\\260 E)', 'ya5f5+lLatitude (\\260 N)'])
    #
    # fig.coast(shorelines="1/0.2p", region=region, projection=pj, water="skyblue")
    # fig.plot(data=gdf.boundary, pen="0.5p,red", projection=pj)
    # fig.show()

    return gdf


def EuropeContinent():
    from src_auxiliary.create_shp import basin2grid_shp
    import geopandas as gpd

    box_area = [71.6, 36.1, -11.1, 42.1]
    gdf = box2shp(box_area=box_area)
    gdf.to_file('/media/user/My Book/Fan/ESA_SING/shapefiles/EuropeContinent/E_box/E_box.shp')

    b2s = basin2grid_shp(grid=(3, 3)).configure(new_basin_name='Europe_subbasins',
                                                original_shp='/media/user/My Book/Fan/ESA_SING/shapefiles/EuropeContinent/E_box')
    b2s.set_modify_func(func=basin2grid_shp.example_func1)
    b2s.create_shp(out_dir='/media/user/My Book/Fan/ESA_SING/shapefiles/EuropeContinent/Europe')

    '''delete invalid subbasins'''
    model_land_mask = '/media/user/My Book/Fan/W3RA_data/basin_selection/model_land_mask.h5'
    GRACE_1deg_land_mask = '/media/user/My Book/Fan/W3RA_data/basin_selection/GlobalLandMaskForGRACE.hdf5'
    Forcing_mask = '/media/user/My Book/Fan/W3RA_data/basin_selection/forcing_mask.npy'

    import h5py
    model = h5py.File(model_land_mask, 'r')['mask'][:-1, :]
    # GRACE = np.flipud(h5py.File(GRACE_1deg_land_mask, 'r')['resolution_1']['mask'][:])
    Forcing = np.load(Forcing_mask)
    mask = model * Forcing
    gdf = gpd.read_file(filename='/media/user/My Book/Fan/ESA_SING/shapefiles/EuropeContinent/Europe')

    invalids = []

    for id in range(1, gdf.ID.size + 1):
        tt = gdf[gdf.ID == id]
        minx = int(float(tt.bounds.minx) / 0.1) + 1800
        maxx = int(float(tt.bounds.maxx) / 0.1) + 1800
        maxy = 900 - int(float(tt.bounds.miny) / 0.1)
        miny = 900 - int(float(tt.bounds.maxy) / 0.1)
        vv = np.sum(mask[miny:maxy + 1, minx:maxx + 1])
        tg = mask[miny:maxy + 1, minx:maxx + 1].size

        flag = True

        '''in case the land area is too small'''
        if vv / tg < 0.16:
            flag = False

        '''in case it is beyond the land mask of GRACE. to be checked manually'''
        '''This only works for this shapefile'''
        if id == 84:
            flag = False

        invalids.append(flag)

        pass

    n = gdf[invalids]
    n.loc[:, 'ID'] = np.arange(np.sum(np.array(invalids))) + 1
    gdf = n

    gdf.to_file('/media/user/My Book/Fan/ESA_SING/shapefiles/EuropeContinent/Europe_valid/Europe_subbasins.shp')
    '''visualization'''
    import pygmt

    fig = pygmt.Figure()
    pygmt.config(MAP_HEADING_OFFSET=0, MAP_TITLE_OFFSET=-0.2)
    pygmt.config(FONT_ANNOT='10p', COLOR_NAN='white')
    pygmt.makecpt(cmap='wysiwyg', series=[0, 56], background='o')

    region = [box_area[2] - 5, box_area[3] + 5, box_area[1] - 5, box_area[0] + 5]
    pj = "Q30/-20/12c"
    fig.basemap(region=region, projection=pj,
                frame=['WSne', 'xa10f5+lLongitude (\\260 E)', 'ya5f5+lLatitude (\\260 N)'])

    fig.coast(shorelines="1/0.2p", region=region, projection=pj, water="skyblue")

    for i in range(1, gdf.shape[0] + 1):
        xy = gdf[gdf.ID == i].centroid
        fig.text(x=xy.x, y=xy.y, text="%s" % i, font='7p,black')

    fig.plot(data=gdf.boundary, pen="0.5p,red", projection=pj)
    fig.show()

    pass


def loadShp():
    import geopandas as gpd
    gdf = gpd.read_file('../data/basin/shp/Europe/Grid_3/Europe_subbasins.shp')

    pass


if __name__ == '__main__':
    # plot_glacier_study()
    # plot_us()
    # USgrid()
    # Africagrid()
    # plot_Africa()
    # Brahmaputra_grid(res=1)
    # Brahmaputra_grid_show(res=5)
    # Brahmaputra_subbasins_show()
    # Danube_grid(res=4)
    # Danube_subbasins_show()
    # Danube_grid_show(res=4)
    # plot_Europe()
    # box2shp()
    EuropeContinent()
    # loadShp()
