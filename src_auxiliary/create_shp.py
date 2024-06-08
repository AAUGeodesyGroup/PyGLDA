import numpy as np
import geopandas as gpd
from shapely import box


def shp_change_Danube():
    import pygmt
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
            self._create_basin_shp(left_corner_point=(left_corner_point_lat[id_basins == tile_id],
                                                      left_corner_point_lon[id_basins == tile_id]),
                                   tile_ID=tile_id + 1)

        pass

    def _create_basin_shp(self, left_corner_point=(90, -180), tile_ID=1):
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

        # lat_rd = lat_lu - sub_basin[0]
        # lon_rd = lon_lu + sub_basin[1]

        # lon_lu, lat_lu = np.meshgrid(lon_lu, lat_lu)
        lon_ru, lat_ru = np.meshgrid(lon_ru, lat_ru)
        lon_ld, lat_ld = np.meshgrid(lon_ld, lat_ld)
        # lon_rd, lat_rd = np.meshgrid(lon_rd, lat_rd)

        poly = box(xmin=lon_ld, ymin=lat_ld, xmax=lon_ru, ymax=lat_ru)
        ID = [i for i in range(1, num_sub_basins + 1)]
        d = {'ID': ID, 'geometry': list(poly.flatten())}
        gdf = gpd.GeoDataFrame(d, crs='epsg:4326')
        gdf.to_file('../res/global_shp_old/Tile%s_subbasins.shp' % tile_ID)

        pass

    def delete_invalid_shp(self):
        """
        delete invalid sub_basins and basins
        """

        '''load data'''
        invalid_subbasin = np.load('../res/invalid_subbasin.npy')

        sub_basin = self.sub_basin

        N = 180 // sub_basin[0]
        M = 360 // sub_basin[1]

        num_sub_basins = N * M
        id_basins = np.arange(num_sub_basins).reshape((N, M))

        N2 = 180 // self.basin[0]
        M2 = 360 // self.basin[1]
        num_basins = N2 * M2

        sub_id_basins = np.vsplit(id_basins, N2)
        sub_id = {}
        i = 0
        for each in sub_id_basins:
            a = np.hsplit(each, M2)
            for m in range(M2):
                sub_id[i + 1] = a[m]
                i += 1

        glaciers = self.removeGlaciers()

        for i in range(1, num_basins + 1):
            if i in glaciers:
                continue

            gf = gpd.read_file('../res/global_shp_old/Tile%s_subbasins.shp' % i)

            x = sub_id[i].flatten()

            a = []
            for m in x:
                if m in invalid_subbasin:
                    a.append(False)
                else:
                    a.append(True)

            if len(gf[a]) == 0:
                continue

            n = gf[a]
            n.loc[:, 'ID'] = np.arange(np.sum(a)) + 1

            n.to_file('../res/global_shp_new/Tile%s_subbasins.shp' % i)
            pass

        pass

    def removeGlaciers(self):

        tile_glacier = [6, 7, 21, 22] + list(range(121, 151))

        return tile_glacier


class global_box_shp_overlap(global_box_shp):
    def __init__(self):
        super().__init__()
        pass

    def configure_extension(self, extension=6):
        self.extension = extension
        assert extension // self.sub_basin[0] * self.sub_basin[0] == extension
        assert extension // self.sub_basin[1] * self.sub_basin[1] == extension
        return self

    def _create_basin_shp(self, left_corner_point=(90, -180), tile_ID=1):
        sub_basin = self.sub_basin

        left_corner_lat = left_corner_point[0] + self.extension
        left_corner_lon = left_corner_point[1] - self.extension

        N = self.basin[0] // sub_basin[0] + self.extension * 2 // sub_basin[0]
        M = self.basin[1] // sub_basin[1] + self.extension * 2 // sub_basin[1]

        # num_sub_basins = N * M

        lat_lu = np.array([left_corner_lat - i * sub_basin[0] for i in range(N)])
        lon_lu = np.array([left_corner_lon + i * sub_basin[1] for i in range(M)])

        lat_lu = lat_lu[lat_lu <= 90]
        lat_lu = lat_lu[lat_lu > -90]
        lon_lu = lon_lu[lon_lu >= -180]
        lon_lu = lon_lu[lon_lu < 180]

        lat_ru = lat_lu.copy()
        lon_ru = lon_lu + sub_basin[1]
        lon_ru = lon_ru[lon_ru <= 180]

        lat_ld = lat_lu - sub_basin[0]
        lon_ld = lon_lu.copy()
        lat_ld = lat_ld[lat_ld >= -90]

        num_sub_basins = len(lon_ru) * len(lat_ru)

        # lat_rd = lat_lu - sub_basin[0]
        # lon_rd = lon_lu + sub_basin[1]

        # lon_lu, lat_lu = np.meshgrid(lon_lu, lat_lu)
        lon_ru, lat_ru = np.meshgrid(lon_ru, lat_ru)
        lon_ld, lat_ld = np.meshgrid(lon_ld, lat_ld)
        # lon_rd, lat_rd = np.meshgrid(lon_rd, lat_rd)

        poly = box(xmin=lon_ld, ymin=lat_ld, xmax=lon_ru, ymax=lat_ru)
        ID = [i for i in range(1, num_sub_basins + 1)]
        d = {'ID': ID, 'geometry': list(poly.flatten())}
        gdf = gpd.GeoDataFrame(d, crs='epsg:4326')
        gdf.to_file('../res/global_shp_overlap/Tile%s_subbasins.shp' % tile_ID)

        pass

    def delete_invalid_shp(self):
        """
        delete invalid sub_basins and basins
        """

        '''load data'''
        invalid_subbasin = np.load('../res/invalid_subbasin.npy')

        '''preparation'''
        sub_basin = self.sub_basin
        N = 180 // sub_basin[0]
        M = 360 // sub_basin[1]
        num_sub_basins = N * M
        id_basins = np.arange(num_sub_basins).reshape((N, M))
        lat = 89.5 - np.arange(N) * sub_basin[0]
        lon = -179.5 + np.arange(M) * sub_basin[1]
        lon, lat = np.meshgrid(lon, lat)

        N2 = 180 // self.basin[0]
        M2 = 360 // self.basin[1]
        num_basins = N2 * M2

        glaciers = self.removeGlaciers()

        sub_id_basins = np.vsplit(id_basins, N2)
        sub_id = {}
        i = 0
        for each in sub_id_basins:
            a = np.hsplit(each, M2)
            for m in range(M2):
                sub_id[i + 1] = a[m]
                i += 1

        '''filter'''
        for i in range(1, num_basins + 1):
            if i in glaciers:
                continue

            gf = gpd.read_file('../res/global_shp_overlap/Tile%s_subbasins.shp' % i)

            bd = gf.unary_union.bounds
            x2 = id_basins[((lat <= bd[3]) * (lat >= bd[1]) * (lon >= bd[0]) * (lon <= bd[2])).astype(bool)].flatten()

            '''judge 1st time'''
            x1 = sub_id[i].flatten()
            a = []
            for m in x1:
                if m in invalid_subbasin:
                    a.append(False)
                else:
                    a.append(True)

            if np.sum(np.array(a)) == 0:
                continue

            '''judge 2nd time'''
            a = []
            for m in x2:
                if m in invalid_subbasin:
                    a.append(False)
                else:
                    a.append(True)

            if len(gf[a]) == 0:
                continue

            n = gf[a]
            n.loc[:, 'ID'] = np.arange(np.sum(a)) + 1

            n.to_file('../res/global_shp_overlap_new/Tile%s_subbasins.shp' % i)
            pass

        pass


class basin2grid_shp:

    def __init__(self, grid=(3, 3)):
        sub_basin = grid

        N = 180 // sub_basin[0]
        M = 360 // sub_basin[1]

        num_sub_basins = N * M

        lat_lu = np.array([90 - i * sub_basin[0] for i in range(N)])
        lon_lu = np.array([-180 + i * sub_basin[1] for i in range(M)])

        lat_ru = lat_lu.copy()
        lon_ru = lon_lu + sub_basin[1]

        lat_ld = lat_lu - sub_basin[0]
        lon_ld = lon_lu.copy()

        # lat_rd = lat_lu - sub_basin[0]
        # lon_rd = lon_lu + sub_basin[1]

        # lon_lu, lat_lu = np.meshgrid(lon_lu, lat_lu)
        lon_ru, lat_ru = np.meshgrid(lon_ru, lat_ru)
        lon_ld, lat_ld = np.meshgrid(lon_ld, lat_ld)
        # lon_rd, lat_rd = np.meshgrid(lon_rd, lat_rd)

        poly = box(xmin=lon_ld, ymin=lat_ld, xmax=lon_ru, ymax=lat_ru)
        ID = [i for i in range(1, num_sub_basins + 1)]
        d = {'ID': ID, 'geometry': list(poly.flatten())}
        self.gdf = gpd.GeoDataFrame(d, crs='epsg:4326')

        pass

    def create_shp(self, new_basin_name, shp):
        import shapely
        from shapely.ops import unary_union
        import pandas as pd

        basin_shp = gpd.read_file(shp)
        basin_all = gpd.GeoDataFrame({'geometry': gpd.GeoSeries(basin_shp.unary_union)}, crs='epsg:4326')

        grid = self.gdf

        index = shapely.intersects(basin_shp.unary_union, grid.geometry).values

        new = []

        for i in np.arange(len(index)):
            if not index[i]:
                continue

            mm = grid[grid.ID == i + 1]
            mm = mm.drop(columns = ['ID'])
            new.append(mm.overlay(basin_all, how='intersection',keep_geom_type=True))

        gdf = gpd.GeoDataFrame(pd.concat(new))
        ID = [i for i in range(1, len(gdf) + 1)]
        d = {'ID': ID, 'geometry': gdf.geometry}
        new_shp = gpd.GeoDataFrame(d, crs='epsg:4326')
        new_shp.to_file('../temp/%s.shp' % new_basin_name)
        pass


def demo1():
    # gbs = global_box_shp().configure_size()
    # gbs.create_shp()

    gbs = global_box_shp_overlap().configure_size().configure_extension(extension=6)
    # gbs.create_shp()
    gbs.delete_invalid_shp()

    pass


def demo2():
    import pygmt
    fig = pygmt.Figure()
    pygmt.config(MAP_HEADING_OFFSET=0, MAP_TITLE_OFFSET=-0.2)
    pygmt.config(FONT_ANNOT='12p', COLOR_NAN='white')
    pygmt.makecpt(cmap='polar', series=[1, 11, 1], background='o')

    region = [90, 160, 10, 40]

    fig.coast(shorelines="1/0.2p", region=region, projection="Q8c")

    gdf = gpd.read_file(filename='../res/global_shp_new/Tile58_subbasins.shp')

    fig.plot(data=gdf.boundary, pen="1p,black")

    # fig.plot(data=gdf[gdf.ID == 8].boundary, cmap=True, fill='blue', pen="1p,black")

    # fig.plot(data=gdf[gdf.ID == 23].boundary, cmap=True, fill='green', pen="1p,black")

    fig.show()

    pass


def demo3():
    import pygmt
    res = 0.1
    err = res / 10
    lat = np.arange(90 - res / 2, -90 + res / 2 - err, -res)
    lon = np.arange(-180 + res / 2, 180 - res / 2 + err, res)
    region = [min(lon), max(lon), min(lat), max(lat)]
    lon, lat = np.meshgrid(lon, lat)

    x = np.load('/media/user/My Book/Fan/W3RA_data/basin_selection/forcing_mask.npy')[:, :].astype(float)
    x[x == 0] = np.nan
    grace = pygmt.xyz2grd(y=lat.flatten(), x=lon.flatten(), z=x.flatten(),
                          spacing=(res, res), region=region)

    fig = pygmt.Figure()
    pygmt.config(MAP_HEADING_OFFSET=0, MAP_TITLE_OFFSET=-0.2)
    pygmt.config(FONT_ANNOT='12p', COLOR_NAN='white')
    pygmt.makecpt(cmap='polar', series=[1, 11, 1], background='o')

    region = [-150, 150, -60, 89.5]

    fig.grdimage(
        grid=grace,
        cmap=True,
        frame=['xa5f5g5', 'ya5f5g5'] + ['+tMask: Danube'],
        dpi=100,
        projection='Q12c',
        region=region,
        interpolation='n'
    )

    fig.coast(shorelines="1/0.2p", region=region, projection="Q12c")

    fig.show()

    pass


def demo4():
    gbs = global_box_shp().configure_size()
    gbs.delete_invalid_shp()
    pass


def demo5():
    import pygmt
    import pandas as pd
    fig = pygmt.Figure()
    pygmt.config(MAP_HEADING_OFFSET=0, MAP_TITLE_OFFSET=-0.2)
    pygmt.config(FONT_ANNOT='7p', COLOR_NAN='white')
    pygmt.makecpt(cmap='polar', series=[1, 11, 1], background='o')

    region = [-180, 180, -90, 90]

    fig.coast(shorelines="1/0.2p", region=region, projection="Q9c", frame=['xa30f15', 'ya30f15'])

    '''new'''
    gdf_list = []
    invalid_basins = []
    for tile in range(1, 150 + 1):
        try:
            gdf = gpd.read_file(filename='../res/global_shp_overlap_new/Tile%s_subbasins.shp' % tile)
            gdf_list.append(gdf)

        except Exception:
            invalid_basins.append(tile)
            continue

    full_gdf = pd.concat(gdf_list)
    fig.plot(data=full_gdf.boundary, pen="0.2p,black", fill='lightgreen', transparency=30)

    '''old'''
    gdf_list = []

    for tile in range(1, 150 + 1):
        if tile in invalid_basins:
            continue
        gdf = gpd.read_file(filename='../res/global_shp_overlap/Tile%s_subbasins.shp' % tile)
        # gdf_list.append(gdf)

        fig.plot(data=gpd.GeoSeries(gdf.unary_union.boundary), pen="0.5p,red", fill='lightblue', transparency=60)
        fig.plot(data=gpd.GeoSeries(gdf.unary_union.boundary), pen="0.5p,red")

    fig.coast(shorelines="1/0.2p", region=region, projection="Q9c")
    fig.show()

    pass


def showbox():
    import pygmt

    fig = pygmt.Figure()

    # Define region of interest
    region = [-11.1, 45.1, 33.9, 76.1]

    # Assign a value of 0 for all water masses and a value of 1 for all land
    # masses.
    # Use shoreline data with (l)ow resolution and set the grid spacing to
    # 5 arc-minutes in x and y direction.
    grid = pygmt.grdlandmask(region=region, spacing="5m", maskvalues=[0, 1], resolution="l")

    # Plot clipped grid
    fig.basemap(region=region, projection="M12c", frame=True)

    # Define a colormap to be used for two categories, define the range of the
    # new discrete CPT using series=(lowest_value, highest_value, interval),
    # use color_model="+cwater,land" to write the discrete color palette
    # "batlow" in categorical format and add water/land as annotations for the
    # colorbar.
    pygmt.makecpt(cmap="batlow", series=(0, 1, 1), color_model="+cwater,land")

    fig.grdimage(grid=grid, cmap=True)
    fig.colorbar(position="JMR+o0.5c/0c+w8c")

    fig.show()

    pass


def demo_show_basin_grid():
    import pygmt
    fig = pygmt.Figure()
    pygmt.config(MAP_HEADING_OFFSET=0, MAP_TITLE_OFFSET=-0.2)
    pygmt.config(FONT_ANNOT='12p', COLOR_NAN='white')
    pygmt.makecpt(cmap='polar', series=[1, 11, 1], background='o')

    region = [0, 35, 40, 60]

    fig.coast(shorelines="1/0.2p", region=region, projection="Q8c")

    gdf = gpd.read_file(filename='../temp/GDRB_subbasins.shp')

    fig.plot(data=gdf.boundary, pen="1p,black")

    fig.show()

    pass


def demo_basin_grid():
    b2g = basin2grid_shp(grid=(3, 3))
    b2g.create_shp(new_basin_name='GDRB_subbasins', shp='../data/basin/shp/DRB_3_shapefiles/DRB_subbasins.shp')

    pass


if __name__ == '__main__':
    # shp_change_Danube()
    # demo1()
    # demo2()
    # demo3()
    # demo4()
    # demo5()
    # showbox()
    demo_basin_grid()
    # demo_show_basin_grid()
