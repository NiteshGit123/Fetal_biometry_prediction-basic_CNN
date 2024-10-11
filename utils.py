import matplotlib.pyplot as plt


def plot_image_with_points(image, points):
    plt.imshow(image.permute(1, 2, 0))
    plt.scatter(points[0::2], points[1::2], c='red', marker='x')
    plt.show()

def display_comparison(original_image, resized_image, original_points, resized_points, image_name):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original_image, cmap='gray')
    plt.scatter(original_points[0::2], original_points[1::2], color='red')
    plt.subplot(1, 2, 2)
    plt.imshow(resized_image, cmap='gray')
    plt.scatter(resized_points[0::2], resized_points[1::2], color='red')
    plt.show()
