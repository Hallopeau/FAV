import rasterio
from rasterio.merge import merge
from rasterio.plot import show
import matplotlib.pyplot as plt

class Merger:
    """
    A class to merge multiple raster images into a single raster image.

    Attributes:
        image_paths: List of paths to the raster images to be merged.
        output_path: Path to save the merged raster image.
    """

    def __init__(self, image_paths, output_path=None):
        """
        Initializes the Merger with the given image paths and output path.

        Args:
            image_paths: List of paths to the raster images.
            output_path: Path to save the merged raster image. Defaults to a combination of input image names.
        """
        self.image_paths = image_paths

        if output_path is None:
            output_path = f"{image_paths[0].split('.')[0].split('/')[-1]}_{image_paths[1].split('.')[0].split('/')[-1]}.tif"
        self.output_path = output_path

    def merge_images(self):
        """
        Merges the raster images specified in image_paths and saves the result to output_path.
        """
        sources = [rasterio.open(path) for path in self.image_paths]

        merged, out_trans = merge(sources)

        merged_meta = sources[0].meta.copy()
        merged_meta.update({
            'transform': out_trans,
            'width': merged.shape[2],
            'height': merged.shape[1]
        })

        with rasterio.open(self.output_path, "w", **merged_meta) as dest:
            dest.write(merged)

    def show_merged_image(self):
        """
        Displays the merged raster image.
        """
        with rasterio.open(self.output_path) as merged_src:
            fig, ax = plt.subplots()
            show(merged_src, ax=ax, transform=merged_src.transform)
            ax.axis('off')
            plt.show()
