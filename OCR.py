import cv2
import numpy as np
import heapq
import pytesseract
from scipy.ndimage import interpolation as inter


def order_points(pts):
    rect = np.zeros((4, 2), dtype='float32')
    pts = np.array(pts)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect.astype('int').tolist()


def find_dest(pts):
    (tl, tr, br, bl) = pts
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    destination_corners = [[0, 0], [maxWidth, 0],
                           [maxWidth, maxHeight], [0, maxHeight]]
    return order_points(destination_corners)


def correct_skew(image, delta=1, limit=5):
    def determine_score(arr, angle):
        data = inter.rotate(arr, angle, reshape=False, order=0)
        histogram = np.sum(data, axis=1)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2)
        return histogram, score

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    scores = []
    angles = np.arange(-limit, limit + delta, delta)
    for angle in angles:
        histogram, score = determine_score(thresh, angle)
        scores.append(score)

    best_angle = angles[scores.index(max(scores))]

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)

    return rotated


def scan(img):
    scale_percent = 0.60
    width = int(img.shape[1] * scale_percent)
    height = int(img.shape[0] * scale_percent)
    img = cv2.resize(img, (width, height))

    orig_img = img.copy()
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (20, 20, img.shape[1] - 20, img.shape[0] - 20)

    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 1, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img = img * mask2[:, :, np.newaxis]

    cv2.namedWindow("Masked Image", cv2.WINDOW_NORMAL)
    cv2.imshow('Masked Image', img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    canny = cv2.Canny(gray, 0, 200)
    canny = cv2.dilate(canny, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (5, 5)))
    cv2.namedWindow("Canny Edges", cv2.WINDOW_NORMAL)
    cv2.imshow('Canny Edges', canny)

    contours, hierarchy = cv2.findContours(
        canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    contour_queue = []
    for i, contour in enumerate(contours):
        heapq.heappush(contour_queue, (cv2.contourArea(contour), i))
        if len(contour_queue) > 2:
            heapq.heappop(contour_queue)

    final_images = []
    for contour_area, contour_index in contour_queue:
        contour = contours[contour_index]
        epsilon = 0.02 * cv2.arcLength(contour, True)
        corners = cv2.approxPolyDP(contour, epsilon, True)
        corners = sorted(np.concatenate(corners).tolist())
        corners = order_points(corners)
        destination_corners = find_dest(corners)

        # Draw contours and corners on the image
        contour_img = cv2.drawContours(
            orig_img.copy(), [contour], -1, (0, 255, 0), 3)
        for corner in corners:
            contour_img = cv2.circle(
                contour_img, tuple(corner), 10, (255, 0, 0), -1)
        cv2.namedWindow("Contours and Corners", cv2.WINDOW_NORMAL)
        cv2.imshow('Contours and Corners', contour_img)

        M = cv2.getPerspectiveTransform(np.float32(
            corners), np.float32(destination_corners))
        final = cv2.warpPerspective(
            orig_img, M, (destination_corners[2][0], destination_corners[2][1]), flags=cv2.INTER_LINEAR)
        final_images.append(final)
        cv2.namedWindow("Final Scanned Image", cv2.WINDOW_NORMAL)
        cv2.imshow('Final Scanned Image', final)
        cv2.waitKey(0)

    return final_images


def ocr_from_image(image_path):
    # Load image
    img = cv2.imread(image_path)
    scanned_imgs = scan(img)

    for i, img in enumerate(scanned_imgs):

        # Correct skew
        corrected = correct_skew(img)

        # Now you can pass this preprocessed image to pytesseract
        result = pytesseract.image_to_string(corrected)

        print(f"Text from scanned image {i + 1}:")
        print("-------------------------")
        print(result)
        print("-------------------------\n")


if __name__ == "__main__":
    # Replace with your image path
    image_path = 'C:/Users/shruj/OneDrive/Documents/EMPEQ/NLP/FSS-A.I/Pre-Processing/images/43_PhotoN_1.jpeg'
    ocr_from_image(image_path)
