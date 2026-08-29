import os

# Target folder
folder = r"D:\Pasien original\20. Ny. Isdartiningsih (36)\20. Ny. Isdartiningsih (36)\IMGDATA\20191029\S0000001"

# Loop through all files inside the folder
for filename in os.listdir(folder):
    if filename.lower().endswith(".dcm"):
        old_path = os.path.join(folder, filename)
        new_name = filename[:-4]  # remove ".dcm"
        new_path = os.path.join(folder, new_name)

        # Rename the file
        os.rename(old_path, new_path)
        print(f"Renamed: {filename} → {new_name}")

print("Done.")
