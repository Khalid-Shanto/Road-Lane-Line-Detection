import cv2
import numpy as np
import matplotlib.pyplot as plt


def region_of_interest(image, vertices):
    """
    Applies an image mask.

    Args:
        image: The input image.
        vertices: A list of vertices of the region of interest.

    Returns:
        The image masked by the region of interest.
    """
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, vertices, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


def detect_lane_lines(image):
    """
    Detects lane lines in an image using computer vision techniques.

    Args:
        image: The input image.

    Returns:
        The image with detected lane lines drawn on it.
    """

    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)



    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    edges = cv2.Canny(blurred_image, 50, 150)

    height, width = edges.shape
    vertices = np.array([[(0, height), (width / 2, height / 2), (width, height)]], dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)

    lines = cv2.HoughLinesP(masked_edges, rho=1, theta=np.pi / 180, threshold=40, minLineLength=50, maxLineGap=100)

    line_image = np.zeros_like(image)

    draw_lines(line_image, lines)

    result = cv2.addWeighted(image, 0.8, line_image, 1, 0)

    return result


def draw_lines(image, lines, color=(255, 0, 0), thickness=5):
    """
    Draws lines on an image.

    Args:
        image: The input image.
        lines: A list of lines.
        color: The color of the lines (default: red).
        thickness: The thickness of the lines (default: 5).
    """
    if lines is None:
        return

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(image, (x1, y1), (x2, y2), color, thickness)


def main():
    # Load the image.
    image = cv2.imread('road.jpg')

    # Detect lane lines.
    result = detect_lane_lines(image)

    # Display the result.
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.show()


if __name__ == '__main__':
    main()
