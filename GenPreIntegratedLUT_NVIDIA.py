import math
import numpy as np
from PIL import Image

# Parameters
image_width = 1024
image_height = 1024
output_path = "Results/CurvatureLUT.png"

# GPUPro2-"Pre-Integrated Skin Shading": Appendix A
def Gaussian(v, r2):  # v->variance
    # G(v,r) = e^(-r^2 / (2v)) / sqrt(2 * pi * v)
    return math.exp(-r2 / (2.0 * v)) / math.sqrt(2.0 * math.pi * v)


# GPUPro2-"Pre-Integrated Skin Shading": Appendix A
def ComputeDiffuseProfile(dist):
    r2 = dist * dist
    # Coefficients from GPU Gems3−"Advanced Skin Rendering"
    return np.array([0.233, 0.455, 0.649]) * Gaussian(0.0064, r2) \
        + np.array([0.100, 0.336, 0.344]) * Gaussian(0.0484, r2) \
        + np.array([0.118, 0.198, 0.000]) * Gaussian(0.1870, r2) \
        + np.array([0.113, 0.007, 0.007]) * Gaussian(0.5670, r2) \
        + np.array([0.358, 0.004, 0.000]) * Gaussian(1.9900, r2) \
        + np.array([0.078, 0.000, 0.000]) * Gaussian(7.4100, r2)


# NVIDIA GTC2014 FaceWorks
def ComputePreintegrateDiffuseOnRing(radius, n_dot_l):
    theta = math.acos(n_dot_l)
    accumulate_diffuse = np.array([0.0, 0.0, 0.0])
    
    range_max = min(math.pi * radius, 10.0)
    range_min = max(-math.pi * radius, -10.0)
    scale = (range_max - range_min) / 200 # 200 Loops
    bias = range_min + 0.5 * scale

    for i in range(200):
        x = i * scale + bias
        diffuse = max(math.cos(theta - x / radius), 0.0)  # saturate
        dist = abs(x)
        profile = ComputeDiffuseProfile(dist)
        accumulate_diffuse += diffuse * profile

    accumulate_diffuse *= scale

    # NVIDIA special process
    adjust = -max(n_dot_l, 0) * 2.0 + 0.5
    return np.clip(accumulate_diffuse * 2.0 + adjust, 0.0, 1.0)


def GenPreintegrateDiffuseImgData():
    image_data = np.zeros((image_width, image_height, 3), dtype=np.uint8)
    
    curvature_scale = 1.0 / image_height # curvature range(0, 1)
    curvature_bias = 0.5 * curvature_scale # start from from 0.0 + 0.5 * scale
    n_dot_l_scale = 2.0 / image_width # n_dot_l range(-1, 1)
    n_dot_l_bias = -1.0 + 0.5 * n_dot_l_scale # start from -1.0 + 0.5 * scale

    for col in range(image_height):
        curvature = col * curvature_scale + curvature_bias
        for row in range(image_width):
            n_dot_l = row * n_dot_l_scale + n_dot_l_bias
            diffuse = ComputePreintegrateDiffuseOnRing(1.0 / curvature, n_dot_l)

            # diffuse = pow(diffuse, 1.0 / 2.2)  # Gamma Correction
            result = diffuse * np.array([255, 255, 255])
            # Flip Y
            image_data[image_height - 1 - col][row] = (
                int(result[0]), int(result[1]), int(result[2]))
    # print(image_data)
    return image_data


# Output to file
lut_img = Image.fromarray(GenPreintegrateDiffuseImgData())
lut_img.save(output_path)
lut_img.show()
