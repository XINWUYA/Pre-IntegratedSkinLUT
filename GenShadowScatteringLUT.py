import math
import numpy as np
from PIL import Image

image_width = 128
image_height = 128
output_path = "ShadowScatteringLUT1.png"

diffusion_radius = 2.7 # in world units (= 2.7mm for human skin)
shadow_width_min = 8 # in world units (mm)
shadow_width_max = 100 # in world units (mm)
shadow_sharpening = 10.0 # Ratio by which output shadow is sharpened (typically 3.0 to 10.0)
shadow_offset = 0.0 # default 0: 


def Gaussian(v, r2):  # v->variance
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


def ComputeShadowScattering(input_pos, penumbra_width):
    accumulate_diffuse = np.array([0.0, 0.0, 0.0])
    accumulate_weight = np.array([0.0, 0.0, 0.0])

    x = -10  # Start from -10
    while x <= 10:
        new_pos = (input_pos + x * penumbra_width) * shadow_sharpening + (0.5 - 0.5 * shadow_sharpening)
        new_pos = np.clip(new_pos, 0.0, 1.0) # saturate
        new_shadow = (3.0 - 2.0 * new_pos) * new_pos * new_pos
        # Calculate R(2r * sin(x/2))
        dist = abs(x)
        profile = ComputeDiffuseProfile(dist)
        accumulate_diffuse += new_shadow * profile
        accumulate_weight += profile
        x += 0.2  # 100 loops
    return accumulate_diffuse / accumulate_weight


# Ref: NVIDIA GTC-2014 FaceWorks
def GenShadowScatteringImgData():
    image_data = np.zeros((image_width, image_height, 3), dtype=np.uint8)

    diffusion_radius_factor = diffusion_radius / 2.7
    rcp_shadow_width_min = diffusion_radius_factor / shadow_width_max
    rcp_shadow_width_max = diffusion_radius_factor / shadow_width_min
    shadow_step_scale = (rcp_shadow_width_max - rcp_shadow_width_min) / image_height

    for row in range(image_height):
        rcp_width = rcp_shadow_width_min + float(row + 0.5) * shadow_step_scale
        for col in range(image_width):
            shadow = float(col + 0.5) / image_width  # (0,1)
            input_pos = (math.sqrt(shadow) - math.sqrt(1.0 + shadow_offset - shadow)) * 0.5 + 0.5 #
            diffuse = ComputeShadowScattering(input_pos, rcp_width)

            # diffuse = pow(diffuse, 1.0 / 2.2)  # Gamma Correction
            result = diffuse * np.array([255, 255, 255])
            # Flip Y
            image_data[image_height - 1 - row][col] = (
                int(result[0]), int(result[1]), int(result[2]))
    # print(image_data)
    return image_data


# Output to file
lut_img = Image.fromarray(GenShadowScatteringImgData())
lut_img.save(output_path)
lut_img.show()