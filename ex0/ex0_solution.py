import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# Do not alter this path!
IMAGE_PATH: str = "data/Image01.png"


class ImageProcessor:
    def __init__(self, image_path: str, colour_type: str = "BGR"):
        """
        Load and save the provided image, the image colour type and the image directory.
        Use CV2 to load the image.

        Args:
        image_path (str): Path to the input image.
        colour_type (str): Colour type of the image (BGR, RGB, Gray).
        """
        # Extract the parent directory of the image.
        self._image_directory: str = os.path.dirname(image_path)
        if colour_type not in ["BGR", "RGB", "Gray"]:
            raise ValueError("The given colour is not supported!")

        # ToDo: Save the colour type and load the image using CV2.
        self._colour_type: str = colour_type
        # self._image: np.ndarray = np.zeros(0)

        #Done
        #Load image in BGR (default)
        image_bgr: np.ndarray = cv2.imread(image_path)

        # Load image in RGB or Gray
        if colour_type == "RGB":
            self._image: np.ndarray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        elif colour_type == "Gray":
            self._image: np.ndarray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        else:  # BGR
            self._image: np.ndarray = image_bgr

    def get_image_data(self):
        return self._image, self._colour_type

    def show_image(self):
        """
        Show the loaded image using either matplotlib or CV2.
        """

        # ToDo: Show the image depending on the colour type.
        # Done
        if self._colour_type == "RGB":
            image_to_show = cv2.cvtColor(self._image, cv2.COLOR_RGB2BGR)
        else:
            image_to_show = self._image

        cv2.imshow("Image", image_to_show)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def save_image(self, image_title: str):
        """
        Save the loaded image using either matplotlib or CV2.

        Args:
        image_title (str): Title of the image with the corresponding extension.
        """

        # Combine the image parent directory and the given title to create the path for the new image.
        total_image_path: str = os.path.join(self._image_directory, image_title)

        # ToDo: Save the image.
        cv2.imwrite(total_image_path, self._image)

    def convert_colour(self):
        """
        Convert a colour image from BGR to RGB or vice versa.
        Do not use functions from external libraries.
        Solve this task by using indexing.
        """
        if self._colour_type not in ["RGB", "BGR"]:
            raise ValueError("The function only works for colour images!")

        # ToDo: Perform the colour conversion.
        self._image = self._image[:, :, ::-1]

        # ToDo: Update the colour type.
        if self._colour_type == "BGR":
            self._colour_type = "RGB"
        else:
            self._colour_type = "BGR"

    def clip_image(self, clip_min: int, clip_max: int):
        """
        Clip all colour values in the image to a given min and max value.
        Do not use functions from external libraries.
        Solve this task by using indexing.

        Args:
        clip_min (int): Minimum image colour intensity.
        clip_max (int): Maximum image colour intensity.
        """
        # ToDo: Clip the image values to the given values.
        # Values smaller than clip_min become clip_min
        self._image[self._image < clip_min] = clip_min
        # Values larger than clip_max become clip_max
        self._image[self._image > clip_max] = clip_max

    def convert_to_grayscale(self, method: str = "lightness"):
        """
        Convert a colour image to a grayscale image.
        Write the different options from scratch.

        Args:
        method (str): Method for the colour conversion, either lightness, average or luminosity.
        """
        if method not in ["lightness", "average", "luminosity"]:
            raise ValueError("The given method is not supported!")
        if self._colour_type not in ["BGR", "RGB"]:
            raise ValueError("The function only works for colour images!")

        if method == "lightness":
            self._image = (
                    self._image.max(axis=2) / 2
                    + self._image.min(axis=2) / 2
            )

        if method == "average":
            self._image = (
                    self._image[:, :, 0] / 3
                    + self._image[:, :, 1] / 3
                    + self._image[:, :, 2] / 3
            )

        if method == "luminosity":
            if self._colour_type == "RGB":
                self._image = (
                        0.21 * self._image[:, :, 0]
                        + 0.72 * self._image[:, :, 1]
                        + 0.07 * self._image[:, :, 2]
                )
            else:  # BGR
                self._image = (
                        0.07 * self._image[:, :, 0]
                        + 0.72 * self._image[:, :, 1]
                        + 0.21 * self._image[:, :, 2]
                )

        # ToDo: Update the colour type.
        self._colour_type = "Gray"

    def rotate_image(self, degrees: int = 0):
        """
        Rotate an image by a given angle (k * 90) clockwise.
        Do not use functions from external libraries apart from numpy.transpose.

        Args:
        degrees (int): Rotation angle.
        """
        if degrees % 90 != 0:
            raise ValueError("The provided rotation angle must be a multiple of 90!")

        # ToDo: Rotate the image depending on the given rotation value.
        degrees = degrees % 360

        if degrees == 0:
            return

        if self._image.ndim == 3:
            if degrees == 90:
                self._image = self._image.transpose(1, 0, 2)[:, ::-1, :]
            elif degrees == 180:
                self._image = self._image[::-1, ::-1, :]
            elif degrees == 270:
                self._image = self._image.transpose(1, 0, 2)[::-1, :, :]

        elif self._image.ndim == 2:
            if degrees == 90:
                self._image = self._image.transpose(1, 0)[:, ::-1]
            elif degrees == 180:
                self._image = self._image[::-1, ::-1]
            elif degrees == 270:
                self._image = self._image.transpose(1, 0)[::-1, :]

    def flip_image(self, flip_value: int):
        """
        Flip an image either vertically (0), horizontally (1) or both ways (2).
        Do not use functions from external libraries.

        Args:
        flip_value (int): Value to determine how the image should be flipped.
        """
        if flip_value not in [0, 1, 2]:
            raise ValueError("The provided flip value must be either 0, 1 or 2!")

        # ToDo: Flip the image using indexing.
        # Vertical flip: slice all elements along the first axis (rows) in reverse
        if flip_value == 0:
            self._image = self._image[::-1]
        # Horizontal flip: slice all elements along the second axis (columns) in reverse
        elif flip_value == 1:
            self._image = self._image[:, ::-1]
        # Horizontal and vertical flip
        else:
            self._image = self._image[::-1, ::-1]

    def crop_center(self, new_height: int, new_width: int):
        """
        Crop the image to a given size around the center.
        Do not use functions from external libraries.

        Args:
        new_height (int): Height of the cropped image.
        new_width (int): Width of the cropped image.
        """
        # ToDo: Check that the given parameters are valid!
        if not isinstance(new_height, int) or not isinstance(new_width, int):
            raise TypeError("New height and new width must be integers!")

        if new_height <= 0 or new_width <= 0:
            raise ValueError("New height and new width must be positive!")

        height = self._image.shape[0]
        width = self._image.shape[1]

        if new_height > height or new_width > width:
            raise ValueError("The cropped image size cannot be larger than the original image!")

        # ToDo: Crop the image around the center.
        start_y = (height - new_height) // 2
        end_y = start_y + new_height

        start_x = (width - new_width) // 2
        end_x = start_x + new_width

        self._image = self._image[start_y:end_y, start_x:end_x]

    def resize_image(self, new_height: int, new_width: int):
        """
        Resize an image to an arbitrary size using CV2.

        Args:
        new_height (int): Height of the resized image.
        new_width (int): Width of the resized image.
        """
        # Done: Check that the given parameters are valid!
        if not isinstance(new_height, int) or not isinstance(new_width, int):
            raise TypeError("New height and new width must be integers!")

        if new_height <= 0 or new_width <= 0:
            raise ValueError("New height and new width must be positive!")

        # ToDo: Resize the image. Research the available options in CV2.
        self._image = cv2.resize(self._image, (new_width, new_height))

if __name__ == '__main__':
    processor = ImageProcessor(image_path=IMAGE_PATH, colour_type="BGR")
    processor.show_image()
