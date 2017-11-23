# For manual override

from utils.config import SaveConfig
import numpy as np

# Common for both camera models
camera_matrix = np.array([
    [11096.77, 0, 540],
    [0, 11096.77, 960],
    [0, 0, 1]
])

# Pinhole
# dist_coeffs = np.array([
#     [-2.57614e-01, 8.7708e-02, -2.5697e-04, -5.933903e-04, -1.5219e-02]
#     ])

# Fisheye
dist_coeffs = np.array([
    [-0.32,  0.126,  0,   0, 0]
])

save_config = SaveConfig('manual_calib', 'calib')
save_config.save(camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
