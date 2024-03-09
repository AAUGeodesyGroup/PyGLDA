import numpy as np


def shp_change_Danube():
    import pygmt
    import geopandas as gpd
    import pandas as pd
    from shapely.ops import unary_union

    fig = pygmt.Figure()
    pygmt.config(MAP_HEADING_OFFSET=0, MAP_TITLE_OFFSET=-0.2)
    pygmt.config(FONT_ANNOT='12p', COLOR_NAN='white')
    pygmt.makecpt(cmap='polar', series=[1, 11, 1], background='o')

    region = [5, 30, 40, 55]

    fig.coast(shorelines="1/0.2p", region=region, projection="Q12c")

    gdf = gpd.read_file('../data/basin/shp/Danube_9_shapefiles/Danube_9_subbasins.shp')

    a = gdf[gdf.ID == 1].geometry
    b = gdf[gdf.ID == 2].geometry
    c = gdf[gdf.ID == 3].geometry
    d = gdf[gdf.ID == 4].geometry
    e = gdf[gdf.ID == 5].geometry
    f = gdf[gdf.ID == 6].geometry
    g = gdf[gdf.ID == 7].geometry
    h = gdf[gdf.ID == 8].geometry
    i = gdf[gdf.ID == 9].geometry

    polygons1 = [a]
    boundary1 = gpd.GeoSeries(unary_union(polygons1))

    polygons2 = [b, c, d, e, f, g]
    boundary2 = gpd.GeoSeries(unary_union(polygons2))

    polygons3 = [h, i]
    boundary3 = gpd.GeoSeries(unary_union(polygons3))

    hjkj = gpd.pd.concat([boundary1, boundary2, boundary3])

    # data = {'ID':[1,2,3], 'geometry':[boundary1.geometry, boundary2.geometry, boundary3.geometry] }
    ds = {'ID': [1, 2, 3]}
    envgdf1 = pd.DataFrame({'geometry': boundary1.geometry, 'ID': 1})
    envgdf2 = pd.DataFrame({'geometry': boundary2.geometry, 'ID': 2})
    envgdf3 = pd.DataFrame({'geometry': boundary3.geometry, 'ID': 3})

    single_df = pd.concat([envgdf1, envgdf2, envgdf3], ignore_index=True).reset_index(drop=True)
    geo_df = gpd.GeoDataFrame(single_df,
                              geometry='geometry',
                              crs='epsg:4326')

    # geo_df.to_file('../res/DRB.shp')
    #
    # geo_df = gpd.read_file('../data/basin/shp/DRB_3_shapefiles')
    # fig.plot(data=geo_df.geometry, pen="1p,black")
    #
    # fig.show()

    pass


import geopandas as gpd
from shapely import box


class global_box_shp:
    def __init__(self):
        self.basin = None
        self.sub_basin = None

    def configure_size(self, sub_basin=(3, 3), basin=(18, 24)):
        self.sub_basin = sub_basin
        self.basin = basin

        for i in range(2):
            assert (basin[i] // sub_basin[i]) * sub_basin[i] == basin[i]

        assert (180 // basin[0]) * basin[0] == 180
        assert (360 // basin[1]) * basin[1] == 360
        return self

    def create_shp(self):
        N = 180 // self.basin[0]
        M = 360 // self.basin[1]

        num_basins = N * M
        id_basins = np.arange(num_basins).reshape((N, M))

        left_corner_point_lat = np.arange(90, -90, -self.basin[0])
        left_corner_point_lon = np.arange(-180, 180, self.basin[1])

        left_corner_point_lon, left_corner_point_lat = np.meshgrid(left_corner_point_lon, left_corner_point_lat)

        assert np.shape(left_corner_point_lon) == np.shape(id_basins)

        for tile_id in range(num_basins):
            print('Tile: %s' % (tile_id + 1))
            self.__create_basin_shp(left_corner_point=(left_corner_point_lat[id_basins == tile_id],
                                                       left_corner_point_lon[id_basins == tile_id]),
                                    tile_ID=tile_id + 1)

        pass

    def __create_basin_shp(self, left_corner_point=(90, -180), tile_ID=1):
        sub_basin = self.sub_basin

        N = self.basin[0] // sub_basin[0]
        M = self.basin[1] // sub_basin[1]

        num_sub_basins = N * M

        lat_lu = np.array([left_corner_point[0] - i * sub_basin[0] for i in range(N)])
        lon_lu = np.array([left_corner_point[1] + i * sub_basin[1] for i in range(M)])

        lat_ru = lat_lu.copy()
        lon_ru = lon_lu + sub_basin[1]

        lat_ld = lat_lu - sub_basin[0]
        lon_ld = lon_lu.copy()

        lat_rd = lat_lu - sub_basin[0]
        lon_rd = lon_lu + sub_basin[1]

        lon_lu, lat_lu = np.meshgrid(lon_lu, lat_lu)
        lon_ru, lat_ru = np.meshgrid(lon_ru, lat_ru)
        lon_ld, lat_ld = np.meshgrid(lon_ld, lat_ld)
        lon_rd, lat_rd = np.meshgrid(lon_rd, lat_rd)

        poly = box(xmin=lon_ld, ymin=lat_ld, xmax=lon_ru, ymax=lat_ru)
        ID = [i for i in range(1, num_sub_basins + 1)]
        d = {'ID': ID, 'geometry': list(poly.flatten())}
        gdf = gpd.GeoDataFrame(d, crs='epsg:4326')
        gdf.to_file('../res/Tile%s_subbasins.shp' % tile_ID)

        pass


def demo1():
    gbs = global_box_shp().configure_size()
    gbs.create_shp()
    pass


def demo2():
    import pygmt
    fig = pygmt.Figure()
    pygmt.config(MAP_HEADING_OFFSET=0, MAP_TITLE_OFFSET=-0.2)
    pygmt.config(FONT_ANNOT='12p', COLOR_NAN='white')
    pygmt.makecpt(cmap='polar', series=[1, 11, 1], background='o')

    region = [90, 140, 10, 40]

    fig.coast(shorelines="1/0.2p", region=region, projection="Q8c")

    gdf = gpd.read_file(filename='../data/basin/shp/globe/Tile58_subbasins.shp')

    fig.plot(data=gdf.boundary, pen="1p,black")

    fig.plot(data=gdf[gdf.ID == 8].boundary, cmap=True, color='blue', pen="1p,black")

    fig.plot(data=gdf[gdf.ID == 23].boundary, cmap=True, color='green', pen="1p,black")

    fig.show()

    pass


if __name__ == '__main__':
    # shp_change_Danube()
    # demo1()
    demo2()
