import geopandas as gpd
import torch

class OutputGenerator:
    """
    A class to generate output shapefiles based on model predictions.

    Attributes:
        model: The trained PyTorch model used for making predictions.
        test_loader: DataLoader for the test dataset.
        dataset: Dataset containing class information.
        grid: GeoDataFrame containing grid data.
        result_path: Path to save the output shapefile.
        device: Device on which to run the model (CPU or GPU).
    """

    def __init__(self, model, test_loader, dataset, grid_file_path, result_path="outputs"):
        """
        Initializes the OutputGenerator with the given parameters.

        Args:
            model: The trained PyTorch model.
            test_loader: DataLoader for the test dataset.
            dataset: Dataset containing class information.
            grid_file_path: Path to the grid file.
            result_path: Path to save the output shapefile.
        """
        self.model = model
        self.test_loader = test_loader
        self.dataset = dataset
        self.grid = gpd.read_file(grid_file_path)
        self.result_path = result_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def create_shapefile(self):
        """
        Generates a shapefile with predicted classes for each grid cell.
        """
        predicted_f = []
        predicted_r = []

        with torch.no_grad():
            for images, labels, img_paths in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)

                for i in range(outputs.shape[0]):
                    output_i = outputs[i]
                    label_i = labels[i]

                    img_path_i = img_paths[i]
                    img_path_i = (img_path_i.split(".")[0]).split("/")[-1]
                    _, predicted = torch.max(output_i.data, 0)
                    
                    if predicted == self.dataset.classes.index("favelas"):
                        predicted_f.append(img_path_i)
                    if predicted == self.dataset.classes.index("residential"):
                        predicted_r.append(img_path_i)

        predicted_f = [int(i) for i in predicted_f]
        predicted_r = [int(i) for i in predicted_r]

        gdf = gpd.GeoDataFrame(geometry=self.grid.geometry)
        gdf["class"] = None
        gdf["id"] = self.grid["id"]

        for index, row in self.grid.iterrows():
            if row["id"] in predicted_f:
                gdf.loc[index, "class"] = "favelas"
            elif row["id"] in predicted_r:
                gdf.loc[index, "class"] = "residential"
            else:
                gdf.loc[index, "class"] = "others"

        gdf.to_file(self.result_path)
        