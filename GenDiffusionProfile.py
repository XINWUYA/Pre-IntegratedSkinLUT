import math
import numpy as np
from PIL import Image

image_width = 128  # 1024
image_height = 128  # 1024
output_path = "DiffusionProfile.png"


def Gaussian(v, r2):
    # G(v,r) = e^(-r^2 / (2v))
    return math.exp(-r2 / (2.0 * v))
    # return math.exp(-r2 / (2.0 * v)) / math.sqrt(2.0 * math.pi * v)

# result may > 1


def ComputeDiffuseProfile(dist):
    r2 = dist * dist
    return np.array([0.233, 0.455, 0.649]) * Gaussian(0.0064, r2) \
        + np.array([0.100, 0.336, 0.344]) * Gaussian(0.0484, r2) \
        + np.array([0.118, 0.198, 0.000]) * Gaussian(0.1870, r2) \
        + np.array([0.113, 0.007, 0.007]) * Gaussian(0.5670, r2) \
        + np.array([0.358, 0.004, 0.000]) * Gaussian(1.9900, r2) \
        + np.array([0.078, 0.000, 0.000]) * Gaussian(7.4100, r2)


def GenDiffuseProfileImgData():
    image_data = np.zeros((image_width, image_height, 3), dtype=np.uint8)
    for col in range(image_height):
        # v = float(col + 0.5) / image_height - 0.5  # (-0.5, 0.5)
        v = float(col) / image_height
        for row in range(image_width):
            # u = float(row + 0.5) / image_width - 0.5  # (-0.5, 0.5)
            # r = 2.0 * math.sqrt(u ** 2 + v ** 2)
            u = float(row) / image_width
            r = 2.0 * math.sqrt((u - 0.5)**2 + (v - 0.5)**2)
            profile = ComputeDiffuseProfile(r)

            # profile = pow(profile, 1.0 / 2.2)  # Gamma Correction
            result = profile * np.array([255, 255, 255])
            result = np.clip(result, 0, 255)  # clamp
            image_data[col][row] = (
                int(result[0]), int(result[1]), int(result[2]))
    return image_data


lut_img = Image.fromarray(GenDiffuseProfileImgData())
lut_img.save(output_path)
lut_img.show()
