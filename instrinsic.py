import numpy as np

# Manaul calculate instrinsic matrix
focal_length_mm = 26
sensor_width_mm = 5.76
sensor_height_mm = 4.29
image_width_px = 4032
image_height_px = 3024 

pixel_size_x = sensor_width_mm / image_width_px
pixel_size_y = sensor_height_mm / image_height_px

focal_length_px_x = focal_length_mm / pixel_size_x
focal_length_px_y = focal_length_mm / pixel_size_y

c_x = image_width_px / 2
c_y = image_height_px / 2

K = np.array([
    [focal_length_px_x, 0, c_x],
    [0, focal_length_px_y, c_y],
    [0, 0, 1]
])

print("Instrinsic matrix")
print(K)

scaled_width = 800
scaled_height = 600

scale_x = scaled_width / image_width_px
scale_y = scaled_height / image_height_px

focal_length_scaled_x = focal_length_px_x * scale_x
focal_length_scaled_y = focal_length_px_y * scale_y

c_x_scaled = c_x * scale_x
c_y_scaled = c_y * scale_y

K_scaled = np.array([
    [focal_length_scaled_x, 0, c_x_scaled],
    [0, focal_length_scaled_y, c_y_scaled],
    [0, 0, 1]
])

print("\nScale instrinsic matrix")
print(K_scaled)

np.set_printoptions(precision=1, suppress=True)
K_round = np.asarray([[K_scaled[0, 0], K_scaled[0, 1], K_scaled[0, 2]],
                 [K_scaled[1, 0], K_scaled[1, 1], K_scaled[1, 2]],
                 [K_scaled[2, 0], K_scaled[2, 1], K_scaled[2, 2]]])
print("\nRound instrinsic matrix")
print(K_round)