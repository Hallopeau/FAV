import geopandas as gpd
import rasterio
from shapely.geometry import Polygon

class Filter:
    """
    A class to filter a grid of polygons based on a raster image.

    Attributes:
        grid_shapefile_path: Path to the shapefile containing the grid.
        image_path: Path to the raster image.
        output_filtered_shapefile_path: Path to save the filtered grid shapefile.
    """

    def __init__(self, grid_shapefile_path, image_path, output_filtered_shapefile_path=None):
        """
        Initializes the Filter with the given parameters.

        Args:
            grid_shapefile_path: Path to the shapefile containing the grid.
            image_path: Path to the raster image.
            output_filtered_shapefile_path: Path to save the filtered grid shapefile. Defaults to the input grid name with a '_f' suffix.
        """
        self.grid_shapefile_path = grid_shapefile_path
        self.image_path = image_path
        if output_filtered_shapefile_path is None:
            output_filtered_shapefile_path = f"{grid_shapefile_path.split('.')[0]}_f.shp"
        self.output_filtered_shapefile_path = output_filtered_shapefile_path

    def filter_grid(self):
        """
        Filters the grid of polygons based on the presence of null pixels in the raster image
        and saves the filtered grid to a shapefile.
        """
        grid_gdf = gpd.read_file(self.grid_shapefile_path)

        with rasterio.open(self.image_path) as src:
            grid_gdf = grid_gdf.to_crs(src.crs)

            indices_to_remove = []

            for idx, poly in grid_gdf.iterrows():
                geom = poly.geometry
                window = src.window(*geom.bounds)
                subset = src.read(window=window)

                if (subset == 0).any():
                    indices_to_remove.append(idx)

        grid_gdf = grid_gdf.drop(indices_to_remove)

        grid_gdf.to_file(self.output_filtered_shapefile_path)
