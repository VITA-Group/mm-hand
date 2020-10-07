import os
import numpy as np
import sys
import struct
from data.base_dataset import BaseDataset
import torch
import torch.utils.data as data
from pyellipsoid import drawing
import cv2
import random

CONNECTIONS = [  # [0, 1],
    [1, 2],
    [2, 3],
    [3, 4],
    # [0, 5],
    [5, 6],
    [6, 7],
    [7, 8],
    # [0, 9],
    [9, 10],
    [10, 11],
    [11, 12],
    # [0, 13],
    [13, 14],
    [14, 15],
    [15, 16],
    # [0, 17],
    [17, 18],
    [18, 19],
    [19, 20]]


def build_rotation_matrix(ax, ay, az, inverse=False):
    """Build a Euler rotation matrix.
    Rotation order is X, Y, Z (right-hand coordinate system).
    Expected vector is [x, y, z].

    Arguments:
        ax {float} -- rotation angle around X (radians)
        ay {float} -- rotation angle around Y (radians)
        az {float} -- rotation angle around Z (radians)

    Keyword Arguments:
        inverse {bool} -- Do inverse rotation (default: {False})

    Returns:
        [numpy.array] -- rotation matrix
    """

    if inverse:
        ax, ay, az = -ax, -ay, -az

    Rx = np.array([[1, 0, 0],
                   [0, np.cos(ax), -np.sin(ax)],
                   [0, np.sin(ax), np.cos(ax)]])

    Ry = np.array([[np.cos(ay), 0, np.sin(ay)],
                   [0, 1, 0],
                   [-np.sin(ay), 0, np.cos(ay)]])

    Rz = np.array([[np.cos(az), -np.sin(az), 0],
                   [np.sin(az), np.cos(az), 0],
                   [0, 0, 1]])

    R = np.dot(Rz, np.dot(Ry, Rx))

    return R


def make_ellipsoid_image(shape, center, radii, angle):
    """Draw a 3D binary image containing a 3D ellipsoid.

    Arguments:
        shape {list} -- image shape [z, y, x]
        center {list} -- center of the ellipsoid [x, y, z]
        radii {list} -- radii [x, y, z]
        angle {list} -- rotation angles [x, y, z]

    Raises:
        ValueError -- arguments are wrong

    Returns:
        [numpy.array] -- image with ellipsoid
    """

    if len(shape) != 3:
        raise ValueError('Only 3D ellipsoids are supported.')

    if not (len(center) == len(radii) == len(shape)):
        raise ValueError('Center, radii of ellipsoid and image shape have different dimensionality.')

    # Do opposite rotation since it is an axes rotation.
    angle = -1 * angle
    R = build_rotation_matrix(*angle)

    # Convert to numpy
    radii = np.array(radii)

    # Build a grid and get its points as a list
    xi = tuple(np.linspace(0, s - 1, s) - np.floor(0.5 * s) for s in shape)

    # Build a list of points forming the grid
    xi = np.meshgrid(*xi, indexing='ij')
    # points = np.array(list(zip(*np.vstack(list(map(np.ravel, xi))))))
    points = np.array(xi).reshape(3, -1)[::-1]

    # Reorder coordinates to match XYZ order and rotate
    # points = points[:, ::-1]
    # points = np.dot(R, points.T).T
    points = np.dot(R, points).T
    # Find grid center and rotate
    grid_center = np.array(center) - 0.5 * np.array(shape[::-1])
    grid_center = np.dot(R, grid_center)

    # Reorder coordinates back to ZYX to match the order of numpy array axis
    points = points[:, ::-1]
    grid_center = grid_center[::-1]
    radii = radii[::-1]

    # Draw the ellipsoid
    # dx**2 + dy**2 + dz**2 = r**2
    # dx**2 / r**2 + dy**2 / r**2 + dz**2 / r**2 = 1
    dR = (points - grid_center) ** 2
    dR = dR / radii ** 2
    # Sum dx, dy, dz / r**2
    nR = np.sum(dR, axis=1).reshape(shape)

    ell = (nR <= 1).astype(np.uint8)

    return ell.T  # [:, ::-1]


def pixel2world(x, y, z, img_width, img_height, fx, fy):
    w_x = (x - img_width / 2) * z / fx
    w_y = (img_height / 2 - y) * z / fy
    w_z = z
    return w_x, w_y, w_z


def world2pixel(x, y, z, img_width, img_height, fx, fy):
    p_x = x * fx / z + img_width / 2
    p_y = img_height / 2 - y * fy / z
    uv = []
    for x, y in zip(p_x, p_y):
        uv.append([x, y])
    return uv


def depthmap2points(image, fx, fy):
    h, w = image.shape
    x, y = np.meshgrid(np.arange(w) + 1, np.arange(h) + 1)
    points = np.zeros((h, w, 3), dtype=np.float32)
    points[:, :, 0], points[:, :, 1], points[:, :, 2] = pixel2world(x, y, image, w, h, fx, fy)
    return points


def points2pixels(points, img_width, img_height, fx, fy):
    pixels = np.zeros((points.shape[0], 2))
    pixels[:, 0], pixels[:, 1] = \
        world2pixel(points[:, 0], points[:, 1], points[:, 2], img_width, img_height, fx, fy)
    return pixels

def get_rotational_value(rf, randomRot, uv, img_width, img_height):
    """
    Re-orientated the hand such that it is always up-right
    :param rf:
    :param randomRot:
    :param uv:
    :param img_width:
    :param img_height:
    :return:
    """
    # ((0, 17), []),
    # ((17, 1), []),
    # ((1, 5), []),
    # ((5, 9), []),
    # ((9, 13), []),
    # ((13, 0), []),
    wrist = np.array(uv[0])
    palms = np.array([uv[1], uv[5], uv[9], uv[13]]).T # the palm excluding thumb
    palms = np.sum(palms, axis=1) / palms.shape[-1]# find centroid point

    center = np.array([img_height/2, img_width/2])
    wrist_norm = wrist - center
    palms_norm = palms - center
    A = (wrist - center) - (palms-center) # vector of hand coordinate
    upside_down = wrist[0] - palms[0] < 0


    A_norm = A / np.linalg.norm(A)
    B_norm = np.array([0, 1])
    theta = np.arccos(np.clip(np.dot(A_norm, B_norm), -1.0, 1.0))
    absdegree = np.rad2deg(theta)

    def direction(start, center , end):
        c = (start[0] - center[0])*(end[1]-center[1]) - (start[1]-center[1])*(end[0]-center[0])
        if c < 0:
            c = -1
        elif c > 0:
            c = 1
        else:
            c = 0
        return -1 * c

    c = direction(wrist, center, [center[0], center[1] + 1])
    if c <= -1:
        degree = c * absdegree
    else:
        degree = absdegree
    # if wrist_norm[0] > palms_norm[0]:
    #     c = direction(wrist, center, [center[0], center[1]+1])
    # else:
    #     c = direction(palms, center, [center[0], center[1]+1])

    return 0



def load_depthmap(filename, img_width, img_height, max_depth, crop_dim, randomRot, uv):
    with open(filename, mode='rb') as f:
        data = f.read()
        _, _, left, top, right, bottom = struct.unpack('I' * 6, data[:6 * 4])
        num_pixel = (right - left) * (bottom - top)
        cropped_image = struct.unpack('f' * num_pixel, data[6 * 4:])
        cropped_image = np.asarray(cropped_image).reshape(bottom - top, -1)
        depth_image = np.zeros((img_height, img_width), dtype=np.float32)
        depth_image[top:bottom, left:right] = cropped_image

        center, scale = xywh2cs(left, top, right - left, bottom - top, crop_dim, crop_dim)
        rf = 25
        uv = np.array(uv)
        rotation = get_rotational_value(rf, 0.5,  uv, img_width, img_height)

        # print(f"rotation level: {rotation}")
        trans = get_affine_transformation(center,
                                          scale,
                                          rot=rotation,
                                          output_size=(crop_dim, crop_dim))
        depth_image[depth_image == 0] = max_depth
        depth_image = cv2.warpAffine(depth_image, trans, (crop_dim, crop_dim))
        depth_image[depth_image == 0] = max_depth

        return depth_image, trans


def get_affine_transformation(center,
                              scale,
                              rot,
                              output_size,
                              shift=np.array([0, 0], dtype=np.float32),
                              inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        print(scale)
        scale = np.array([scale, scale])

    scale_tmp = scale * 200.0
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def xywh2cs(x, y, w, h, img_width, img_height):
    center = np.zeros((2), dtype=np.float32)
    center[0] = x + w * 0.5
    center[1] = y + h * 0.5
    aspect_ratio = img_width * 1. / img_height
    pixel_std = 200
    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array(
        [w * 1.0 / pixel_std, h * 1.0 / pixel_std],
        dtype=np.float32)
    if center[0] != -1:
        scale = scale * 1.25

    return center, scale


def discretize(coord, cropped_size):
    '''[-1, 1] -> [0, cropped_size]'''
    min_normalized = -1
    max_normalized = 1
    scale = (max_normalized - min_normalized) / cropped_size
    return (coord - min_normalized) / scale


def warp2continuous(coord, refpoint, cubic_size, cropped_size):
    '''
    Map coordinates in set [0, 1, .., cropped_size-1] to original range [-cubic_size/2+refpoint, cubic_size/2 + refpoint]
    '''
    min_normalized = -1
    max_normalized = 1

    scale = (max_normalized - min_normalized) / cropped_size
    coord = coord * scale + min_normalized  # -> [-1, 1]

    coord = coord * cubic_size / 2 + refpoint

    return coord


def scattering(coord, cropped_size):
    # coord: [0, cropped_size]
    # Assign range[0, 1) -> 0, [1, 2) -> 1, .. [cropped_size-1, cropped_size) -> cropped_size-1
    # That is, around center 0.5 -> 0, around center 1.5 -> 1 .. around center cropped_size-0.5 -> cropped_size-1
    coord = coord.astype(np.int32)

    mask = (coord[:, 0] >= 0) & (coord[:, 0] < cropped_size) & \
           (coord[:, 1] >= 0) & (coord[:, 1] < cropped_size) & \
           (coord[:, 2] >= 0) & (coord[:, 2] < cropped_size)

    coord = coord[mask, :]

    cubic = np.zeros((cropped_size, cropped_size, cropped_size))

    # Note, directly map point coordinate (x, y, z) to index (i, j, k), instead of (k, j, i)
    # Need to be consistent with heatmap generating and coordinates extration from heatmap
    cubic[coord[:, 0], coord[:, 1], coord[:, 2]] = 1

    return cubic


def extract_coord_from_output(output, center=True):
    '''
    output: shape (batch, jointNum, volumeSize, volumeSize, volumeSize)
    center: if True, add 0.5, default is true
    return: shape (batch, jointNum, 3)
    '''
    assert (len(output.shape) >= 3)
    vsize = output.shape[-3:]

    output_rs = output.reshape(-1, np.prod(vsize))
    max_index = np.unravel_index(np.argmax(output_rs, axis=1), vsize)
    max_index = np.array(max_index).T

    xyz_output = max_index.reshape([*output.shape[:-3], 3])

    # Note discrete coord can represents real range [coord, coord+1), see function scattering()
    # So, move coord to range center for better fittness
    if center: xyz_output = xyz_output + 0.5

    return xyz_output


def generate_coord(points, refpoint, new_size, angle, trans, sizes):
    cubic_size, cropped_size, original_size = sizes

    # points shape: (n, 3)
    coord = points

    # note, will consider points within range [refpoint-cubic_size/2, refpoint+cubic_size/2] as candidates

    # normalize
    coord = (coord - refpoint) / (cubic_size / 2)  # -> [-1, 1]
    # print(f"refpoint: {refpoint}")
    # print(f"coord: {coord}")

    # discretize
    coord = discretize(coord, cropped_size)  # -> [0, cropped_size]
    coord += (original_size / 2 - cropped_size / 2)  # move center to original volume
    # print(f"coord norm: {coord}")

    # resize around original volume center
    resize_scale = new_size / 100
    if new_size < 100:
        coord = coord * resize_scale + original_size / 2 * (1 - resize_scale)
    elif new_size > 100:
        coord = coord * resize_scale - original_size / 2 * (resize_scale - 1)
    else:
        # new_size = 100 if it is in test mode
        pass

    # rotation
    if angle != 0:
        original_coord = coord.copy()
        original_coord[:, 0] -= original_size / 2
        original_coord[:, 1] -= original_size / 2
        coord[:, 0] = original_coord[:, 0] * np.cos(angle) - original_coord[:, 1] * np.sin(angle)
        coord[:, 1] = original_coord[:, 0] * np.sin(angle) + original_coord[:, 1] * np.cos(angle)
        coord[:, 0] += original_size / 2
        coord[:, 1] += original_size / 2

    # translation
    # Note, if trans = (original_size/2 - cropped_size/2), the following translation will
    # cancel the above translation(after discretion). It will be set it when in test mode.
    coord -= trans

    return coord


def generate_cubic_input(points, refpoint, new_size, angle, trans, sizes):
    _, cropped_size, _ = sizes
    coord = generate_coord(points, refpoint, new_size, angle, trans, sizes)

    # scattering
    cubic = scattering(coord, cropped_size)

    return cubic


def generate_cubic_hand(keypoints, refpoint, new_size, angle, trans, sizes):
    _, cropped_size, _ = sizes
    coord = generate_coord(keypoints, refpoint, new_size, angle, trans, sizes)

    # scattering
    cubic = scattering(coord.copy(), cropped_size)

    # adding shape
    # centers = []
    # raddis = []
    for i, j in CONNECTIONS:
        j1, j2 = coord[i], coord[j]

        r_x = abs(j1[0] - j2[0])
        r_y = abs(j1[1] - j2[1])
        r_z = abs(j1[2] - j2[2])
        d = np.sqrt(r_x ** 2 + r_y ** 2 + r_z ** 2)

        rot = np.deg2rad([80, 30, 20])

        c_x = abs(j1[0] - j2[0]) / 2 + min(j1[0], j2[0])
        c_y = abs(j1[1] - j2[1]) / 2 + min(j1[1], j2[1])
        c_z = abs(j1[2] - j2[2]) / 2 + min(j1[2], j2[2])

        # centers.append((c_x, c_y, c_z))

        cubic += make_ellipsoid_image(cubic.shape, (c_x, c_y, c_z), [d / 2.5] * 3, rot)
    cubic = (cubic >= 1).astype(np.float32)
    return cubic


def generate_heatmap_gt(keypoints, refpoint, new_size, angle, trans, sizes, d3outputs, pool_factor, std):
    _, cropped_size, _ = sizes
    d3output_x, d3output_y, d3output_z = d3outputs

    coord = generate_coord(keypoints, refpoint, new_size, angle, trans, sizes)  # [0, cropped_size]
    coord /= pool_factor  # [0, cropped_size/pool_factor]

    # heatmap generation
    output_size = int(cropped_size / pool_factor)
    heatmap = np.zeros((keypoints.shape[0], output_size, output_size, output_size))

    # use center of cell
    center_offset = 0.5

    for i in range(coord.shape[0]):
        xi, yi, zi = coord[i]
        heatmap[i] = np.exp(-(np.power((d3output_x + center_offset - xi) / std, 2) / 2 + \
                              np.power((d3output_y + center_offset - yi) / std, 2) / 2 + \
                              np.power((d3output_z + center_offset - zi) / std, 2) / 2))

    return heatmap


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def generate_jointsmap(uv_coord, depth, width, height, channel=3):
    canvas = np.zeros((height, width))
    bones = [
        ((17, 18), [130] * channel),
        ((18, 19), [140] * channel),
        ((19, 20), [150] * channel),

        ((1, 2), [10] * channel),
        ((2, 3), [20] * channel),
        ((3, 4), [30] * channel),

        ((5, 6), [40] * channel),
        ((6, 7), [50] * channel),
        ((7, 8), [60] * channel),

        ((9, 10), [70] * channel),
        ((10, 11), [80] * channel),
        ((11, 12), [90] * channel),

        ((13, 14), [100] * channel),
        ((14, 15), [110] * channel),
        ((15, 16), [120] * channel),

        ((0, 17), []),
        ((0, 1), []),
        ((0, 5), []),
        ((0, 9), []),
        ((0, 13), [])
    ]
    palm = []
    for connection, _ in [((0, 17), []),
                          ((17, 1), []),
                          ((1, 5), []),
                          ((5, 9), []),
                          ((9, 13), []),
                          ((13, 0), []), ]:
        coord1 = uv_coord[connection[0]]
        palm.append([int(coord1[0]), int(coord1[1])])
    # palm.append([int((coord1[0]-.5)* W_scale+ W_offset ), int(-(coord1[1]- .5)* H_scale+ H_offset)])
    # print(palm)
    palm_colors = [depth[0], depth[17], depth[1],  depth[5], depth[9], depth[13]]
    palm_colors = list(filter(lambda x: x >= 0, palm_colors))
    # if len(palm_colors) == 6:
    #     binary_mask = np.zeros((height, width))
    #     cv2.fillConvexPoly(binary_mask, np.array([palm], dtype=np.int32), 1)
    #
    #     avg_color_upper = np.average(palm_colors[2::])
    #     avg_color_lower = np.average(palm_colors[::2])
    #
    #     Xs, Ys = np.array(palm).T
    #     Xmax, Xmin = np.max(Xs), np.min(Xs)
    #     Ymax, Ymin = np.max(Ys), np.min(Ys)
    #
    #     orientation = None
    #     s_c = None # start_color
    #     d_c = None # end_color
    #     wrist = np.array([palm[0], palm[1]]).T
    #     if Xmax in wrist[0]:
    #         orientation = "leftRight"
    #         s_c = avg_color_upper
    #         d_c = avg_color_lower
    #     elif Xmin in wrist[0]:
    #         orientation = "RightLeft"
    #         s_c = avg_color_lower
    #         d_c = avg_color_upper
    #     elif Ymax in wrist[1]:
    #         orientation = "TopDown"
    #         s_c = avg_color_upper
    #         d_c = avg_color_lower
    #     else:
    #         orientation = "BottomUp"
    #         s_c = avg_color_lower
    #         d_c = avg_color_upper
    #
    #     n_step = Xmax - Xmin if 'left' in orientation.lower() else Ymax - Ymin
    #
    #     gradient_offset = np.abs(avg_color_lower - avg_color_upper) / n_step
    #
    #     def add(x, y):
    #         return x+y
    #     def minus(x,y):
    #         return x-y
    #
    #     color_operation = minus if s_c > d_c else add
    #
    #     for i in range(int(n_step)):
    #         s_c = color_operation(s_c, i * gradient_offset)
    #         if 'left' in orientation.lower():
    #             canvas[:, Xmin + i] = s_c
    #         else:
    #             canvas[Ymin +i, :] = s_c
    #
    #     canvas = np.multiply(canvas, binary_mask)
    # else:
    if len(palm_colors):
        pass
        # cv2.fillPoly(canvas, np.array([palm], dtype=np.int32), np.average(palm_colors))



    for connection, color in bones:
        temp_canvas = np.zeros(canvas.shape)
        coord1 = uv_coord[connection[0]]
        coord2 = uv_coord[connection[1]]
        coords = np.stack([coord1, coord2])
        colors = [depth[connection[0]], depth[connection[1]]]
        if -1 in colors or len(colors) == 0:
            continue
        else:
            color = np.average(colors)
        # 0.5, 0.5 is the center
        x = coords[:, 0]
        y = coords[:, 1]
        mX = x.mean()
        mY = y.mean()
        length = ((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2) ** 0.5
        angle = np.math.degrees(np.math.atan2(y[0] - y[1], x[0] - x[1]))
        radius = 2
        polygon = cv2.ellipse2Poly((int(mX), int(mY)), (int(length / 3), radius), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(temp_canvas, polygon, color)
        canvas = np.maximum(canvas, temp_canvas)

    return canvas


def normalize(image, min_value, max_value):
    # image = cv2.normalize(image, None, alpha=1.0, norm_type=cv2.NORM_MINMAX)
    # inverted_image = np.abs(image - 1.0)
    image = (image - min_value) / (max_value - min_value)
    image[image < 0] = 0
    return image


def gaussian_kernel(height, width, x, y, sigma):
    gridy, gridx = np.mgrid[0:width, 0:height]
    D2 = (gridx - x) ** 2 + (gridy - y) ** 2
    return np.exp(-D2 / 2.0 / sigma / sigma)


def generate_heatmap(x, y, width, height, sigma, depth, max_depth=700):
    # centermap = np.zeros((height, width, 1), dtype=np.float32)
    center_map = gaussian_kernel(width, height, x, y, sigma)
    # print(center_map.shape)
    center_map[center_map > 1] = 1
    center_map[center_map < 0.0099] = 0
    center_map *= depth
    # centermap[:, :, 0] = center_map

    return center_map


class V2VVoxelization(object):
    def __init__(self, cubic_size, augmentation=True):
        self.cubic_size = cubic_size
        self.cropped_size, self.original_size = 64, 96
        self.sizes = (self.cubic_size, self.cropped_size, self.original_size)
        self.pool_factor = 2
        self.std = 1.7
        self.augmentation = augmentation

        output_size = int(self.cropped_size / self.pool_factor)
        # Note, range(size) and indexing = 'ij'
        self.d3outputs = np.meshgrid(np.arange(output_size), np.arange(output_size), np.arange(output_size),
                                     indexing='ij')

    def __call__(self, sample):
        points, keypoints, refpoint = sample['points'], sample['keypoints'], sample['refpoint']

        ## Augmentations
        # Resize
        new_size = np.random.rand() * 40 + 80

        # Rotation
        angle = np.random.rand() * 80 / 180 * np.pi - 40 / 180 * np.pi

        # Translation
        trans = np.random.rand(3) * (self.original_size - self.cropped_size)

        if not self.augmentation:
            new_size = 100
            angle = 0
            trans = self.original_size / 2 - self.cropped_size / 2

        input = generate_cubic_input(points, refpoint, new_size, angle, trans, self.sizes)
        keypointsVoxel = generate_cubic_hand(keypoints, refpoint, new_size, angle, trans, self.sizes)
        # keypointsVoxel = generate_cubic_input(keypoints, refpoint, new_size, angle, trans, self.sizes)

        keypoints = generate_coord(keypoints, refpoint, new_size, angle, trans, self.sizes)
        # heatmap = generate_heatmap_gt(keypoints, refpoint, new_size, angle, trans, self.sizes, self.d3outputs,
        #                               self.pool_factor, self.std)

        return input.reshape((1, *input.shape)), keypoints, keypointsVoxel.reshape(
            (1, *keypointsVoxel.shape))  # , heatmap

    def voxelize(self, points, refpoint):
        new_size, angle, trans = 100, 0, self.original_size / 2 - self.cropped_size / 2
        input = generate_cubic_input(points, refpoint, new_size, angle, trans, self.sizes)
        return input.reshape((1, *input.shape))

    def generate_heatmap(self, keypoints, refpoint):
        new_size, angle, trans = 100, 0, self.original_size / 2 - self.cropped_size / 2
        heatmap = generate_heatmap_gt(keypoints, refpoint, new_size, angle, trans, self.sizes, self.d3outputs,
                                      self.pool_factor, self.std)
        return heatmap

    def evaluate(self, heatmaps, refpoints):
        coords = extract_coord_from_output(heatmaps)
        coords *= self.pool_factor
        keypoints = warp2continuous(coords, refpoints, self.cubic_size, self.cropped_size)
        return keypoints


class MSRAHandDataset(BaseDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument("--num_keypoints", type=int, default=21)
        parser.add_argument('--centerdir', type=str, default=f"datasets/msra_center")
        return parser

    def __init__(self, opt):
        super(MSRAHandDataset, self).__init__(opt)
        self.img_width = 320
        self.img_height = 240
        self.min_depth = 100
        self.max_depth = 700
        self.fx = 241.42
        self.fy = 241.42
        self.joint_num = 21
        self.world_dim = 3
        self.folder_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9',
                            'I', 'IP', 'L', 'MP', 'RP', 'T', 'TIP', 'Y']
        self.subject_num = 9

        self.root = opt.dataroot
        self.center_dir = f"datasets/msra_center"
        self.test_subject_id = 3
        self.mode = 'train' if opt.isTrain else 'test'
        self.transform = V2VVoxelization(200, True)

        self.updatable_rot = 0.6
        self.step_rot = 0.05

        if not self.mode in ['train', 'test']: raise ValueError('Invalid mode')
        assert self.test_subject_id >= 0 and self.test_subject_id < self.subject_num

        if not self._check_exists(): raise RuntimeError('Invalid MSRA hand dataset')

        self._load()

    def __getitem__(self, index):
        crop_dim = 256
        xyz = self.joints_world[index]
        z = xyz[:, -1]
        uv = world2pixel(xyz[:, 0], xyz[:, 1], xyz[:, 2], self.img_width, self.img_height, self.fx, self.fy)

        depthmap, trans = load_depthmap(self.names[index], self.img_width, self.img_height, self.max_depth, crop_dim, self.updatable_rot, uv)
        # points = depthmap2points(depthmap, self.fx, self.fy)
        # points = points.reshape((-1, 3))

        for i, pair in enumerate(uv):
            uv[i] = affine_transform(pair, trans)

        # depth inversion
        depthmap = np.ones(depthmap.shape) * 700.0 - depthmap
        z = np.ones(z.shape) * 700 - z
        max_value = 700
        min_value = 0

        heatmaps_image = np.zeros((crop_dim, crop_dim))
        heatmaps = []
        ordermap = np.zeros((crop_dim, crop_dim))
        z_norms = []
        for i, (x, y) in enumerate(uv):
            if x >= crop_dim or y >= crop_dim or x < 0 or y < 0:
                z_norms.append(-1)
                heatmaps.append(np.zeros(depthmap.shape))
                continue
            z_value = depthmap[int(y), int(x)] if depthmap[int(y), int(x)] > 0 else z[i]
            z_norm = (z_value - min_value) / (max_value - min_value)
            z_norms.append(z_norm)

            gaussian_map = generate_heatmap(x, y, crop_dim, crop_dim, 2.5, 1)

            heatmaps.append(gaussian_map)
            heatmaps_image = np.maximum(gaussian_map * z_norm, heatmaps_image)

        jointsmap = np.squeeze(generate_jointsmap(uv, z_norms, crop_dim, crop_dim, 1))
        heatmaps_image = np.maximum(heatmaps_image, jointsmap)

        heatmaps = np.stack(heatmaps)
        z_norms = [[i] for i in z_norms]
        sample = {
            # 'name': self.names[index],
            'depthmap': normalize(depthmap, min_value, max_value),
            'heatmaps': heatmaps_image,
            'gaussian_pts': heatmaps,
            'refpoint': self.ref_pts[index],
            'fx': self.fx,
            'fy': self.fy,
            'trans': trans,
            'uv': uv,
            'z': z_norms
        }

        sample = self._transform(sample)

        return sample

    def __len__(self):
        return self.num_samples

    def _load(self):
        self._compute_dataset_size()

        self.num_samples = self.train_size if self.mode == 'train' else self.test_size
        self.joints_world = np.zeros((self.num_samples, self.joint_num, self.world_dim))
        self.ref_pts = np.zeros((self.num_samples, self.world_dim))
        self.names = []

        # Collect reference center points strings
        if self.mode == 'train':
            ref_pt_file = 'center_train_' + str(self.test_subject_id) + '_refined.txt'
        else:
            ref_pt_file = 'center_test_' + str(self.test_subject_id) + '_refined.txt'

        with open(os.path.join(self.center_dir, ref_pt_file)) as f:
            ref_pt_str = [l.rstrip() for l in f]

        #
        file_id = 0
        frame_id = 0

        for mid in range(self.subject_num):
            if self.mode == 'train':
                model_chk = (mid != self.test_subject_id)
            elif self.mode == 'test':
                model_chk = (mid == self.test_subject_id)
            else:
                raise RuntimeError('unsupported mode {}'.format(self.mode))

            if model_chk:
                for fd in self.folder_list:
                    annot_file = os.path.join(self.root, 'P' + str(mid), fd, 'joint.txt')

                    lines = []
                    with open(annot_file) as f:
                        lines = [line.rstrip() for line in f]

                    # skip first line
                    for i in range(1, len(lines)):
                        # referece point
                        splitted = ref_pt_str[file_id].split()
                        if splitted[0] == 'invalid':
                            print('Warning: found invalid reference frame')
                            file_id += 1
                            continue
                        else:
                            self.ref_pts[frame_id, 0] = float(splitted[0])
                            self.ref_pts[frame_id, 1] = float(splitted[1])
                            self.ref_pts[frame_id, 2] = float(splitted[2])

                        # joint point
                        splitted = lines[i].split()
                        for jid in range(self.joint_num):
                            self.joints_world[frame_id, jid, 0] = float(splitted[jid * self.world_dim])
                            self.joints_world[frame_id, jid, 1] = float(splitted[jid * self.world_dim + 1])
                            self.joints_world[frame_id, jid, 2] = -float(splitted[jid * self.world_dim + 2])

                        filename = os.path.join(self.root, 'P' + str(mid), fd, '{:0>6d}'.format(i - 1) + '_depth.bin')
                        self.names.append(filename)

                        frame_id += 1
                        file_id += 1

    def _compute_dataset_size(self):
        self.train_size, self.test_size = 0, 0

        for mid in range(self.subject_num):
            num = 0
            for fd in self.folder_list:
                annot_file = os.path.join(self.root, 'P' + str(mid), fd, 'joint.txt')
                with open(annot_file) as f:
                    num = int(f.readline().rstrip())
                if mid == self.test_subject_id:
                    self.test_size += num
                else:
                    self.train_size += num

    def _check_exists(self):
        # Check basic data
        for mid in range(self.subject_num):
            for fd in self.folder_list:

                annot_file = os.path.join(self.root, 'P' + str(mid), fd, 'joint.txt')
                # print(f"fd: {fd}")
                # print(f"annofile: {annot_file}")
                if not os.path.exists(annot_file):
                    print('Error: annotation file {} does not exist'.format(annot_file))
                    return False

        # Check precomputed centers by v2v-hand model's author
        for subject_id in range(self.subject_num):
            center_train = os.path.join(self.center_dir, 'center_train_' + str(subject_id) + '_refined.txt')
            center_test = os.path.join(self.center_dir, 'center_test_' + str(subject_id) + '_refined.txt')
            if not os.path.exists(center_train) or not os.path.exists(center_test):
                print('Error: precomputed center files do not exist')
                return False

        return True

    def _transform(self, sample):
        for k, v in sample.items():
            sample[k] = torch.tensor(v, dtype=torch.float32)
            if k in ['depthmap', 'heatmaps']:
                sample[k] = torch.unsqueeze(sample[k], dim=0)
        return sample
