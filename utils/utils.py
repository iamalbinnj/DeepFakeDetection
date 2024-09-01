# List of packages to include in the requirements.txt file
packages = [
    "tensorflow",
    "matplotlib",
    "pandas",
    "numpy",
    "imageio",
    "opencv-python",
    "IPython",
    "scikit-learn"
]

# Write the package names to the requirements.txt file
with open('requirements.txt', 'w') as f:
    for package in packages:
        f.write(f"{package}\n")

print("requirements.txt file has been created.")
