import geopandas as gpd
import rasterio
from shapely.geometry import Polygon

class GridGenerator:
    """
    A class to generate a grid of polygons over a raster image.

    Attributes:
        image_path: Path to the raster image.
        taille_carreau: Size of each grid cell in the same units as the raster's CRS.
        output_grid_path: Path to save the generated grid shapefile.
    """

    def __init__(self, image_path, output_grid_path=None, taille_carreau=150):
        """
        Initializes the GridGenerator with the given parameters.

        Args:
            image_path: Path to the raster image.
            output_grid_path: Path to save the generated grid shapefile. Defaults to the same name as the image with a .shp extension.
            taille_carreau: Size of each grid cell in the same units as the raster's CRS.
        """
        self.image_path = image_path
        self.taille_carreau = taille_carreau
        if output_grid_path is None:
            output_grid_path = f"{image_path.split('.')[0]}.shp"
        self.output_grid_path = output_grid_path

    def generate_grid(self):
        """
        Generates a grid of polygons over the raster image and saves it to a shapefile.
        """
        with rasterio.open(self.image_path) as src:
            bounds = src.bounds

        # Define the size of the grid cell
        xmin, ymin, xmax, ymax = bounds
        rows = int((ymax - ymin) / self.taille_carreau)
        cols = int((xmax - xmin) / self.taille_carreau)
        polys = []

        for i in range(cols):
            for j in range(rows):
                # Calculate the geographic coordinates of the polygon corners
                x_left = xmin + i * self.taille_carreau
                x_right = xmin + (i + 1) * self.taille_carreau
                y_top = ymax - j * self.taille_carreau
                y_bottom = ymax - (j + 1) * self.taille_carreau

                # Create the polygon with the geographic coordinates
                poly = Polygon([(x_left, y_top), (x_right, y_top), (x_right, y_bottom), (x_left, y_bottom)])
                polys.append(poly)

        grid_gdf = gpd.GeoDataFrame(geometry=polys, crs=src.crs)

        grid_gdf.to_file(self.output_grid_path)
