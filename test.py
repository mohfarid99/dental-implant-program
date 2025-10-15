import numpy as np
import os

# The path to the .npy file as provided by the user.
FILE_PATH = r"D:\[3] Dataset Final [IMAGE and SAVED STATE ONLY]\2_Ramadhan_12_LOIS SETIAWATI\Implant Dimension\final_AB.npy"

def print_npy_shape(file_path):
    """
    Loads an NPY file from the given path and prints its shape.

    Args:
        file_path (str): The full path to the .npy file.
    """
    print(f"Attempting to load file: {file_path}")

    # Check if the file exists before attempting to load
    if not os.path.exists(file_path):
        print("\nERROR: File not found!")
        print("Please ensure the path is correct and the file exists.")
        return

    try:
        # Load the array data from the .npy file
        data = np.load(file_path)

        # Print the shape (dimensions) of the loaded NumPy array
        print("\nSuccessfully loaded the file.")
        print(f"Data type (dtype): {data.dtype}")
        print(f"Array Shape (dimensions): {data.shape}")

    except Exception as e:
        # Catch other potential errors, like permission issues or corrupted file
        print(f"\nAn error occurred while loading the file: {e}")

if __name__ == "__main__":
    print_npy_shape(FILE_PATH)
