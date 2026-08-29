import cv2
import numpy as np
import matplotlib.pyplot as plt


def skeletonize_with_steps(image_path, show=True):
    # --- Load image as grayscale ---
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Image not found or cannot be read")

    # --- Binarize ---
    _, I = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # --- Structuring element: cross 3x3 ---
    K = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    # --- Initialize ---
    S = np.zeros_like(I)   # Skeleton
    iteration = 0
    current = I.copy()

    while True:
        iteration += 1

        # Erosion
        E = cv2.erode(current, K)

        # Stop condition
        if cv2.countNonZero(E) == 0:
            break

        # Dilation of eroded image
        D = cv2.dilate(E, K)

        # Skeleton component
        T = cv2.subtract(current, D)

        # Accumulate skeleton
        S = cv2.bitwise_or(S, T)

        if show:
            fig, axs = plt.subplots(1, 5, figsize=(18, 4))
            axs[0].imshow(current, cmap='gray')
            axs[0].set_title(f"I (Iter {iteration})")

            axs[1].imshow(E, cmap='gray')
            axs[1].set_title("E = I ⊖ K")

            axs[2].imshow(D, cmap='gray')
            axs[2].set_title("D = E ⊕ K")

            axs[3].imshow(T, cmap='gray')
            axs[3].set_title("T = I − D")

            axs[4].imshow(S, cmap='gray')
            axs[4].set_title("S = S ∪ T")

            for ax in axs:
                ax.axis("off")

            plt.tight_layout()
            plt.show()

        # Prepare for next iteration
        current = E.copy()

    return S


# ==========================
# Example usage
# ==========================
if __name__ == "__main__":
    skeleton = skeletonize_with_steps("skeleton.png", show=True)
