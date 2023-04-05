
# This utility file is a modified version of the following code:
# https://github.com/Elucidation/ChessboardDetect/blob/master/FindChessboards.py

#
# Imports
#
import PIL.Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
np.set_printoptions(suppress=True, linewidth=200)  # Better formatting
plt.rcParams['image.cmap'] = 'jet'  # Default colormap is jet

#
# Helpers
#


# Saddles


def getSaddle(gray_img):
    # https://en.wikipedia.org/wiki/Sobel_operator
    img = gray_img.astype(np.float64)
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    gxx = cv2.Sobel(gx, cv2.CV_64F, 1, 0)
    gyy = cv2.Sobel(gy, cv2.CV_64F, 0, 1)
    gxy = cv2.Sobel(gx, cv2.CV_64F, 0, 1)

    S = gxx*gyy - gxy**2
    return S


def nonmax_sup(img, win=10):
    w, h = img.shape
#     img = cv2.blur(img, ksize=(5,5))
    img_sup = np.zeros_like(img, dtype=np.float64)
    for i, j in np.argwhere(img):
        # Get neigborhood
        ta = max(0, i-win)
        tb = min(w, i+win+1)
        tc = max(0, j-win)
        td = min(h, j+win+1)
        cell = img[ta:tb, tc:td]
        val = img[i, j]
        if np.sum(cell.max() == cell) > 1:
            print(cell.argmax())
        if cell.max() == val:
            img_sup[i, j] = val
    return img_sup


def pruneSaddle(s):
    thresh = 128
    score = (s > 0).sum()
    while (score > 10000):
        thresh = thresh*2
        s[s < thresh] = 0
        score = (s > 0).sum()


def getMinSaddleDist(saddle_pts, pt):
    best_dist = None
    best_pt = pt
    for saddle_pt in saddle_pts:
        saddle_pt = saddle_pt[::-1]
        dist = np.sum((saddle_pt - pt)**2)
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_pt = saddle_pt
    return best_pt, np.sqrt(best_dist)


# Contours


def simplifyContours(contours):
    for i in range(len(contours)):
        # Approximate contour and update in place
        # https://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm
        contours[i] = cv2.approxPolyDP(
            contours[i], 0.04*cv2.arcLength(contours[i], True), True)


def is_square(cnt, eps=3.0, xratio_thresh=0.5):
    # 4x2 array, rows are each point, columns are x and y
    center = cnt.sum(axis=0)/4

    # Side lengths of rectangular contour
    dd0 = np.sqrt(((cnt[0, :] - cnt[1, :])**2).sum())
    dd1 = np.sqrt(((cnt[1, :] - cnt[2, :])**2).sum())
    dd2 = np.sqrt(((cnt[2, :] - cnt[3, :])**2).sum())
    dd3 = np.sqrt(((cnt[3, :] - cnt[0, :])**2).sum())

    # diagonal ratio
    xa = np.sqrt(((cnt[0, :] - cnt[2, :])**2).sum())
    xb = np.sqrt(((cnt[1, :] - cnt[3, :])**2).sum())
    xratio = xa/xb if xa < xb else xb/xa

    # Check whether all points part of convex hull
    # ie. not this http://i.stack.imgur.com/I6yJY.png
    # all corner angles, angles are less than 180 deg, so not necessarily internal angles
    ta = getAngle(dd3, dd0, xb)
    tb = getAngle(dd0, dd1, xa)
    tc = getAngle(dd1, dd2, xb)
    td = getAngle(dd2, dd3, xa)
    angle_sum = np.round(ta+tb+tc+td)

    is_convex = np.abs(angle_sum - 360) < 5

    angles = np.array([ta, tb, tc, td])
    good_angles = np.all((angles > 40) & (angles < 140))

    # side ratios
    dda = dd0 / dd1
    if dda < 1:
        dda = 1. / dda
    ddb = dd1 / dd2
    if ddb < 1:
        ddb = 1. / ddb
    ddc = dd2 / dd3
    if ddc < 1:
        ddc = 1. / ddc
    ddd = dd3 / dd0
    if ddd < 1:
        ddd = 1. / ddd
    side_ratios = np.array([dda, ddb, ddc, ddd])
    good_side_ratios = np.all(side_ratios < eps)

    # Return whether side ratios within certain ratio < epsilon
    return (
        # abs(1.0 - dda) < eps and
        # abs(1.0 - ddb) < eps and
        # xratio > xratio_thresh and
        # good_side_ratios and
        # is_convex and
        good_angles)


def getAngle(a, b, c):
    # Get angle given 3 side lengths, in degrees
    k = (a*a+b*b-c*c) / (2*a*b)
    # Handle floating point errors
    if (k < -1):
        k = -1
    elif k > 1:
        k = 1
    return np.arccos(k) * 180.0 / np.pi


def getContourVals(cnt, img):
    cimg = np.zeros_like(img)
    cv2.drawContours(cimg, [cnt], 0, color=255, thickness=-1)
    return img[cimg != 0]


def pruneContours(contours, hierarchy, saddle):
    new_contours = []
    new_hierarchies = []
    for i in range(len(contours)):
        cnt = contours[i]
        h = hierarchy[i]

        # Must be child
        if h[2] != -1:
            continue

        # Only rectangular contours allowed
        if len(cnt) != 4:
            continue

        # Only contours that fill an area of at least 8x8 pixels
        if cv2.contourArea(cnt) < 8*8:
            continue

        if not is_square(cnt):
            continue

        # TODO : Remove those where internal luma variance is greater than threshold

        cnt = updateCorners(cnt, saddle)
        # If not all saddle corners
        if len(cnt) != 4:
            continue

        new_contours.append(cnt)
        new_hierarchies.append(h)
    new_contours = np.array(new_contours)
    new_hierarchy = np.array(new_hierarchies)
    if len(new_contours) == 0:
        return new_contours, new_hierarchy

    # Prune contours below median area
    areas = [cv2.contourArea(c) for c in new_contours]
    mask = [areas >= np.median(areas)*0.25] and [areas <= np.median(areas)*2.0]
    new_contours = new_contours[mask]
    new_hierarchy = new_hierarchy[mask]
    return np.array(new_contours), np.array(new_hierarchy)


def getContours(img, edges, iters=10):
    # Morphological Gradient to get internal squares of canny edges.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
    # MORPH_GRADIENT means The result will look like the outline of the object.
    edges_gradient = cv2.morphologyEx(edges, cv2.MORPH_GRADIENT, kernel)

    # https://docs.opencv.org/3.1.0/d4/d73/tutorial_py_contours_begin.html
    contours, hierarchy = cv2.findContours(
        edges_gradient, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    contours = list(contours)

    simplifyContours(contours)

    return np.array(contours,  dtype=object), hierarchy[0]


# Corners


def updateCorners(contour, saddle):
    ws = 4  # half window size (+1)
    new_contour = contour.copy()
    for i in range(len(contour)):
        cc, rr = contour[i, 0, :]
        rl = max(0, rr-ws)
        cl = max(0, cc-ws)
        window = saddle[rl:min(saddle.shape[0], rr+ws+1),
                        cl:min(saddle.shape[1], cc+ws+1)]
        br, bc = np.unravel_index(window.argmax(), window.shape)
        s_score = window[br, bc]
        br -= min(ws, rl)
        bc -= min(ws, cl)
        if s_score > 0:
            new_contour[i, 0, :] = cc+bc, rr+br
        else:
            return []
    return new_contour


# Grid


def getIdentityGrid(N):
    a = np.arange(N)
    b = a.copy()
    aa, bb = np.meshgrid(a, b)
    return np.vstack([aa.flatten(), bb.flatten()]).T


def getChessGrid(quad):
    quadA = np.array([[0, 1], [1, 1], [1, 0], [0, 0]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(quadA, quad.astype(np.float32))
    quadB = getIdentityGrid(4)-1
    quadB_pad = np.pad(quadB, ((0, 0), (0, 1)), 'constant', constant_values=1)
    C_thing = (np.matrix(M)*quadB_pad.T).T
    C_thing[:, :2] /= C_thing[:, 2]
    return C_thing


def findGoodPoints(grid, spts, max_px_dist=5):
    # Snap grid points to closest saddle point within range and return updated grid = Nx2 points on grid
    new_grid = grid.copy()
    chosen_spts = set()
    N = len(new_grid)
    grid_good = np.zeros(N, dtype=np.bool)
    def hash_pt(pt): return "%d_%d" % (pt[0], pt[1])

    for pt_i in range(N):
        pt2, d = getMinSaddleDist(spts, grid[pt_i, :2].A.flatten())
        if hash_pt(pt2) in chosen_spts:
            d = max_px_dist
        else:
            chosen_spts.add(hash_pt(pt2))
        if (d < max_px_dist):  # max dist to replace with
            new_grid[pt_i, :2] = pt2
            grid_good[pt_i] = True
    return new_grid, grid_good


def getInitChessGrid(quad):
    quadA = np.array([[0, 1], [1, 1], [1, 0], [0, 0]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(quadA, quad.astype(np.float32))
    return makeChessGrid(M, 1)


def makeChessGrid(M, N=1):
    ideal_grid = getIdentityGrid(2+2*N)-N
    ideal_grid_pad = np.pad(ideal_grid, ((0, 0), (0, 1)),
                            'constant', constant_values=1)  # Add 1's column
    # warped_pts = M*pts
    grid = (np.matrix(M)*ideal_grid_pad.T).T
    grid[:, :2] /= grid[:, 2]  # normalize by t
    grid = grid[:, :2]  # remove 3rd column
    return grid, ideal_grid, M


def generateNewBestFit(grid_ideal, grid, grid_good):
    a = np.float32(grid_ideal[grid_good])
    b = np.float32(grid[grid_good])
    M = cv2.findHomography(a, b, cv2.RANSAC)
    return M


def getGrads(img):
    img = cv2.blur(img, (5, 5))
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1)

    grad_mag = gx*gx+gy*gy
    grad_phase = np.arctan2(gy, gx)  # from -pi to pi
    grad_phase_masked = grad_phase.copy()
    gradient_mask_threshold = 2*np.mean(grad_mag.flatten())
    grad_phase_masked[grad_mag < gradient_mask_threshold] = np.nan
    return grad_mag, grad_phase_masked, grad_phase, gx, gy


def getBestLines(img_warped):
    grad_mag, grad_phase_masked, grad_phase, gx, gy = getGrads(img_warped)

    # X
    gx_pos = gx.copy()
    gx_pos[gx_pos < 0] = 0
    gx_neg = -gx.copy()
    gx_neg[gx_neg < 0] = 0
    score_x = np.sum(gx_pos, axis=0) * np.sum(gx_neg, axis=0)
    # Y
    gy_pos = gy.copy()
    gy_pos[gy_pos < 0] = 0
    gy_neg = -gy.copy()
    gy_neg[gy_neg < 0] = 0
    score_y = np.sum(gy_pos, axis=1) * np.sum(gy_neg, axis=1)

    # Choose best internal set of 7
    a = np.array([(offset + np.arange(7) + 1) *
                 32 for offset in np.arange(1, 11-2)])
    scores_x = np.array([np.sum(score_x[pts]) for pts in a])
    scores_y = np.array([np.sum(score_y[pts]) for pts in a])

    # 15x15 grid, so along an axis a set of 7, and an internal 7 at that, so 13x13 grid, 7x7 possibility inside
    # We're also using a 1-padded grid so 17x17 grid
    # We only want the internal choices (13-7) so 6x6 possible options in the 13x13
    # so 2,3,4,5,6,7,8 to 8,9,10,11,12,13,14 ignoring 0,1 and 15,16,17
    best_lines_x = a[scores_x.argmax()]
    best_lines_y = a[scores_y.argmax()]
    return (best_lines_x, best_lines_y)


# Image processing


def loadImage(filepath, resolution=1000.0):
    img_orig = PIL.Image.open(filepath)

    img_width, img_height = img_orig.size

    # Resize
    aspect_ratio = min(resolution/img_width, resolution/img_height)
    new_width, new_height = (
        (np.array(img_orig.size) * aspect_ratio)).astype(int)
    img = img_orig.resize((new_width, new_height), resample=PIL.Image.BILINEAR)

    img_grey = img.convert('L')  # grayscale
    img_grey = np.array(img_grey)

    img_color = img.convert('RGB')  # color
    img_color = np.array(img_color)

    return img_grey, img_color


def findChessboard(img, min_pts_needed=15, max_pts_needed=25):
    blur_img = cv2.blur(img, (3, 3))  # Blur the image
    saddle = getSaddle(blur_img)  # Get the saddle points
    saddle = -saddle  # Invert the saddle points
    saddle[saddle < 0] = 0  # Remove negative values
    pruneSaddle(saddle)  # Prune the saddle points
    s2 = nonmax_sup(saddle)  # Non-maximum suppression
    s2[s2 < 100000] = 0  # Remove small values
    spts = np.argwhere(s2)  # Get the saddle points

    edges = cv2.Canny(img, 20, 250)  # Get the edges (Canny edge detection)
    contours_all, hierarchy = getContours(img, edges)  # Get the contours
    contours, hierarchy = pruneContours(
        contours_all, hierarchy, saddle)  # Prune the contours

    curr_num_good = 0  # Current number of good points
    curr_grid_next = None
    curr_grid_good = None
    curr_M = None

    for cnt_i in range(len(contours)):
        cnt = contours[cnt_i].squeeze()
        grid_curr, ideal_grid, M = getInitChessGrid(cnt)

        for grid_i in range(7):
            grid_curr, ideal_grid, _ = makeChessGrid(M, N=(grid_i+1))
            grid_next, grid_good = findGoodPoints(grid_curr, spts)
            num_good = np.sum(grid_good)
            if num_good < 4:
                M = None
                break
            M, _ = generateNewBestFit(ideal_grid, grid_next, grid_good)
            if M is None or np.abs(M[0, 0] / M[1, 1]) > 15 or np.abs(M[1, 1] / M[0, 0]) > 15:
                M = None
                print("Failed to converge on this one")
                break
        if M is None:
            continue
        elif num_good > curr_num_good:
            curr_num_good = num_good
            curr_grid_next = grid_next
            curr_grid_good = grid_good
            curr_M = M

        # If we found something with more than max needed, good enough to stop here
        if num_good > max_pts_needed:
            break

    # If we found something
    if curr_num_good > min_pts_needed:
        final_ideal_grid = getIdentityGrid(2+2*7)-7
        return curr_M, final_ideal_grid, curr_grid_next, curr_grid_good, spts
    else:
        return None, None, None, None, None


def getUnwarpedPoints(best_lines_x, best_lines_y, M):
    x, y = np.meshgrid(best_lines_x, best_lines_y)
    xy = np.vstack([x.flatten(), y.flatten()]).T.astype(np.float32)
    xy = np.expand_dims(xy, 0)

    xy_unwarp = cv2.perspectiveTransform(xy, M)
    return xy_unwarp[0, :, :]

def getBoardOutline(best_lines_x, best_lines_y, M):
    d = best_lines_x[1] - best_lines_x[0]
    ax = [best_lines_x[0]-d, best_lines_x[-1]+d]
    ay = [best_lines_y[0]-d, best_lines_y[-1]+d]
    x, y = np.meshgrid(ax, ay)
    xy = np.vstack([x.flatten(), y.flatten()]).T.astype(np.float32)
    xy = xy[[0, 1, 3, 2, 0], :]
    xy = np.expand_dims(xy, 0)

    xy_unwarp = cv2.perspectiveTransform(xy, M)
    return xy_unwarp[0, :, :]


# To warp a single point given the homography matrix H
# https://stackoverflow.com/questions/57399915/how-do-i-determine-the-locations-of-the-points-after-perspective-transform-in-t
# spts is a 2D numpy array of saddle points, H is the homography matrix
def warpPoints(pts, H):
    
    # Array to store the warped points
    pts_warped = np.zeros(pts.shape)

    # Iterate through the saddle points
    for i in range(len(pts)):
        
        p = pts[i]
        
        # x coordinate of warped point
        pts_warped[i][0] = (H[0][0]*p[0] + H[0][1]*p[1] + H[0][2]) / \
            ((H[2][0]*p[0] + H[2][1]*p[1] + H[2][2]))
        
        # y coordinate of warped point
        pts_warped[i][1] = (H[1][0]*p[0] + H[1][1]*p[1] + H[1][2]) / \
            ((H[2][0]*p[0] + H[2][1]*p[1] + H[2][2]))

    return pts_warped

def warp_image(filename, plot=False):

    # Load the image and find the chessboard on it
    img, img_orig = loadImage(filename)
    M, ideal_grid, grid_next, grid_good, spts = findChessboard(
        img)  # M is the warp matrix

    # Check the call succeeded (Warp matrix is not none)
    if M is not None:

        # Generate a mapping to warp (Crop) the image
        M, _ = generateNewBestFit((ideal_grid+8)*32, grid_next, grid_good)
        img_warp = cv2.warpPerspective(
            img, M, (17*32, 17*32), flags=cv2.WARP_INVERSE_MAP)

        # Find the best lines that cut the board
        best_lines_x, best_lines_y = getBestLines(img_warp)

        # Get the unwarped points
        inner_corners_unwarped = getUnwarpedPoints(best_lines_x, best_lines_y, M)
        board_outline_unwarp = getBoardOutline(best_lines_x, best_lines_y, M)

        side_len = 2048
        pts_dest = np.array(
            [[0, side_len], [side_len, side_len], [side_len, 0], [0, 0]])

        # Calculate homography matrix and use it to warp the image
        h, _ = cv2.findHomography(board_outline_unwarp[0:4], pts_dest)
        im_out = cv2.warpPerspective(
            np.squeeze(img_orig), h, (side_len, side_len))
        
        # Warp the corners of the board using the homography matrix
        inner_corners_warped = warpPoints(inner_corners_unwarped, h)
        
        # Plot if required
        if plot:
            
            # Plot the original image with the detected corners, best lines, and outer contour
            plt.figure(frameon=False, figsize=(20, 20))
            imshow(img_orig, cmap='Greys_r')
            plt.plot(board_outline_unwarp[:, 0], board_outline_unwarp[:, 1], 'ro-', markersize=5, linewidth=5)
            plt.plot(grid_next[grid_good, 0].A, grid_next[grid_good, 1].A, 'gs', markersize=12)
            # for line in best_lines_x:
            #     plt.axvline(line, color='red', lw=2)
            # for line in best_lines_y:
            #     plt.axhline(line, color='green', lw=2)
            ax = plt.gca()
            ax.set_axis_off()

            # Plot the final image with the warped corners
            plt.figure(frameon=False, figsize=(30, 30))
            imshow(im_out, cmap='Greys_r', aspect='auto')
            plt.plot(inner_corners_warped[:, 0],
                     inner_corners_warped[:, 1], 'bo', markersize=30)  # Plot the detected corners of the image for display
            ax = plt.gca()
            ax.set_axis_off()
            
        return im_out

    # If the call didn't succeed, print an error message
    else:
        print(f'Could not preprocess: {filename}')
        return None

def crop_individual_squares(warped_image):

    # Extract the image's width and height
    image_height = int(warped_image.shape[0])
    image_width = int(warped_image.shape[1])

    # Round down to the nearest multiple of 8
    image_height = image_height - (image_height % 8)
    image_width = image_width - (image_width % 8)

    # Compute the side length of a single square
    square_height = image_height // 8
    square_width = image_width // 8

    # The side of a square will be the minimum of the height and width
    square_side = min(square_height, square_width)

    # Define an array to hold the resulting image's individual squares
    # The squares are indexed from 0 to 63, with 0 being a1, 1 being a2, ..., 8 being b1, ..., 63 being h8
    squares = []

    # To keep track of square label
    for i in range(8):
        for j in range(8):
            square = warped_image[
                (8-j-1)*square_side:(8-j)*square_side, i*square_side:(i+1)*square_side]
            square = cv2.resize(square, (130, 130)) # We use an aspect ratio of 130 * 130 for an individual square
            squares.append(square)

    # Return the array of squares
    return squares