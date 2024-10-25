import numpy


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


if __name__ == '__main__':
    # plot_glacier_study()
    # plot_us()
    # USgrid()
    # Africagrid()
    plot_Africa()
