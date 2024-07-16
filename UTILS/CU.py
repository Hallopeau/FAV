import geopandas as gpd
import rasterio
from rasterio.windows import Window
import os

class Cutter:
    def __init__(self, vector_file, raster_file, output_folder):
        """
        Initializes the Cutter class with vector and raster files, and the output folder.

        Parameters:
        vector_file (str): Path to the vector file.
        raster_file (str): Path to the raster file.
        output_folder (str): Folder where the cut images will be saved.
        """
        self.vector_file = vector_file
        self.raster_file = raster_file
        self.output_folder = output_folder
        
    def class_carreau(self, row, p_fav):
        """
        Determines the class of a tile based on several criteria.

        Parameters:
        row (GeoSeries): A row from the GeoDataFrame containing tile information.
        p_fav (float): Threshold to classify a tile as favelas.

        Returns:
        str: The class of the tile (residential, favelas, vegetation, others).
        """
        if row["res"] == 1.0 and row["p_vegeta"] < 0.95 and row["ghsl"] > 0.5 and row["zi"] == 0.0:
            return "residential"
        if row["res"] == 0.0 and row["p_favelas"] >= p_fav and row["p_vegeta"] < 0.95 and row["ghsl"] > 0.5 and row["zi"] == 0.0:
            return "favelas"
        if row["p_vegeta"] >= 0.95:
            return "vegetation"
        return "others"
        
    def cut_images(self, p_fav):
        """
        Cuts the raster images based on the geometries from the vector file and classifies them.

        Parameters:
        p_fav (float): Threshold to classify a tile as favelas.
        """
        couche_vecteur = gpd.read_file(self.vector_file)

        gdf = gpd.GeoDataFrame(geometry=couche_vecteur.geometry)
        gdf["class"] = None
        gdf["id"] = couche_vecteur["id"]

        with rasterio.open(self.raster_file) as src:
            for index, row in couche_vecteur.iterrows():
                identifiant_carreau = row["id"]
                classe_carreau = self.class_carreau(row, p_fav)
                geom = row.geometry

                gdf.loc[index, "class"] = classe_carreau

                window = src.window(*geom.bounds)
                subset = src.read(window=window)

                profile = src.profile
                profile.update({
                    "height": window.height,
                    "width": window.width,
                    "transform": rasterio.windows.transform(window, src.transform)
                })

                destination_folder = os.path.join(self.output_folder, classe_carreau.lower())
                os.makedirs(destination_folder, exist_ok=True)

                cut_image_path = os.path.join(destination_folder, f"{identifiant_carreau}.tif")
                with rasterio.open(cut_image_path, "w", **profile) as dst:
                    dst.write(subset)
                    
        gdf.to_file(os.path.join(self.output_folder, "check"))
