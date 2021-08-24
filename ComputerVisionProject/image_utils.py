import cv2
import numpy as np
import matplotlib.pyplot as plt

def draw_region(img, vertices_l, vertices_r):
    line_color_l = (0, 0, 255)
    line_color_r = (255, 0, 0)
    thickness = 2
    
    image = np.copy(img)
    
    image = cv2.line(image, vertices_l[0], vertices_l[1], line_color_l, thickness)
    image = cv2.line(image, vertices_l[1], vertices_l[2], line_color_l, thickness)
    image = cv2.line(image, vertices_l[2], vertices_l[3], line_color_l, thickness)
    image = cv2.line(image, vertices_l[3], vertices_l[0], line_color_l, thickness)

    image = cv2.line(image, vertices_r[0], vertices_r[1], line_color_r, thickness)
    image = cv2.line(image, vertices_r[1], vertices_r[2], line_color_r, thickness)
    image = cv2.line(image, vertices_r[2], vertices_r[3], line_color_r, thickness)
    image = cv2.line(image, vertices_r[3], vertices_r[0], line_color_r, thickness)
    return image

def draw_region_gray(img, vertices_l, vertices_r):
    line_color_l = (255)
    line_color_r = (255)
    thickness = 2
    
    image = np.copy(img)
    
    image = cv2.line(image, vertices_l[0], vertices_l[1], line_color_l, thickness)
    image = cv2.line(image, vertices_l[1], vertices_l[2], line_color_l, thickness)
    image = cv2.line(image, vertices_l[2], vertices_l[3], line_color_l, thickness)
    image = cv2.line(image, vertices_l[3], vertices_l[0], line_color_l, thickness)

    image = cv2.line(image, vertices_r[0], vertices_r[1], line_color_r, thickness)
    image = cv2.line(image, vertices_r[1], vertices_r[2], line_color_r, thickness)
    image = cv2.line(image, vertices_r[2], vertices_r[3], line_color_r, thickness)
    image = cv2.line(image, vertices_r[3], vertices_r[0], line_color_r, thickness)
    return image

def get_left_right_regions_color(image, corners_l, corners_r):
    img_l_color = image[corners_l[0][1]:corners_l[2][1], corners_l[0][0]:corners_l[2][0], :]
    img_r_color = image[corners_r[0][1]:corners_r[2][1], corners_r[0][0]:corners_r[2][0], :]
    return img_l_color, img_r_color

def get_left_right_regions_gray(image, corners_l, corners_r):
    img_l_gray = image[corners_l[0][1]:corners_l[2][1], corners_l[0][0]:corners_l[2][0]]
    img_r_gray = image[corners_r[0][1]:corners_r[2][1], corners_r[0][0]:corners_r[2][0]]
    return img_l_gray, img_r_gray

def show_left_right_rgb(image, img_l, img_r):
    f, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(24, 9))

    ax0.set_title('Image', fontsize=20)
    ax1.set_title('Left', fontsize=20)
    ax2.set_title('Right', fontsize=20)

    ax0.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax1.imshow(cv2.cvtColor(img_l, cv2.COLOR_BGR2RGB))
    ax2.imshow(cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB))

def show_left_right_gray(image_gray, img_l_gray, img_r_gray):
    f, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(24, 9))

    ax0.set_title('Image', fontsize=20)
    ax1.set_title('Left', fontsize=20)
    ax2.set_title('Right', fontsize=20)

    ax0.imshow(image_gray, cmap='gray')
    ax1.imshow(img_l_gray, cmap='gray')
    ax2.imshow(img_r_gray, cmap='gray')