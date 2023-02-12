import math
import numpy as np
from PIL import Image


class SSSMethod:
    PreintegrateSSS = 0
    SeparableSSS = 1


# Parameters
image_width = 1024
image_height = 1024
output_path = "PreintegratedLUT.png"
sss_method = SSSMethod.PreintegrateSSS
# sss_method = SSSMethod.SeparableSSS
falloff_color = np.array([1.0, 0.3, 0.2])

if sss_method == SSSMethod.PreintegrateSSS:
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
else:
    # https://www.shadertoy.com/view/NdBGDz
    def Gaussian(v, r2):
        return math.exp(-r2 / (2.0 * v)) / (2.0 * math.pi * v)

    def ComputeDiffuseProfile(dist):
        r2 = (dist / (0.001 + falloff_color)) ** 2
        return 0.233 * np.array([Gaussian(0.0064, r2[0]), Gaussian(0.0064, r2[1]), Gaussian(0.0064, r2[2])]) \
            + 0.100 * np.array([Gaussian(0.0484, r2[0]), Gaussian(0.0484, r2[1]), Gaussian(0.0484, r2[2])]) \
            + 0.118 * np.array([Gaussian(0.1870, r2[0]), Gaussian(0.1870, r2[1]), Gaussian(0.1870, r2[2])]) \
            + 0.113 * np.array([Gaussian(0.5670, r2[0]), Gaussian(0.5670, r2[1]), Gaussian(0.5670, r2[2])]) \
            + 0.358 * np.array([Gaussian(1.9900, r2[0]), Gaussian(1.9900, r2[1]), Gaussian(1.9900, r2[2])]) \
            + 0.078 * np.array([Gaussian(7.4100, r2[0]),
                               Gaussian(7.4100, r2[1]), Gaussian(7.4100, r2[2])])


# GPUPro2-"Pre-Integrated Skin Shading": D(theta, r)
def ComputePreintegrateDiffuseOnRing(radius, n_dot_l):
    # theta = math.acos(2.0 * n_dot_l - 1.0) # theta in (0, pi)
    theta = math.pi * (1 - n_dot_l)  # instead method
    accumulate_diffuse = np.array([0.0, 0.0, 0.0])
    accumulate_weight = np.array([0.0, 0.0, 0.0])

    # x = -math.pi * 0.5  # Start from -pi / 2
    # while x <= math.pi * 0.5:
    x = -math.pi  # Start from -pi
    while x <= math.pi:
        diffuse = max(math.cos(theta + x), 0.0)  # saturate
        # Calculate R(2r * sin(x/2))
        dist = abs(2.0 * radius * math.sin(0.5 * x))
        profile = ComputeDiffuseProfile(dist)
        accumulate_diffuse += diffuse * profile
        accumulate_weight += profile
        x += 0.01 * math.pi  # 100 loops
    return accumulate_diffuse / accumulate_weight


def GenPreintegrateDiffuseImgData():
    image_data = np.zeros((image_width, image_height, 3), dtype=np.uint8)
    for col in range(image_height):
        radius = float(col + 0.5) / image_height  # (0,1)
        for row in range(image_width):
            n_dot_l = float(row + 0.5) / image_width  # (0,1)
            diffuse = ComputePreintegrateDiffuseOnRing(1.0 / radius, n_dot_l)

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
