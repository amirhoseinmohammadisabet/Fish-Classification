import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Directory Path
directory_path = "Data/Fish_Data/images/cropped/"

# Count Labels
cnt = {}
for filename in os.listdir(directory_path):
    if os.path.isfile(os.path.join(directory_path, filename)):
        name = filename.split("_")[0]
        cnt[name] = cnt.get(name, 0) + 1

print(f"Number of labels: {len(cnt)}")

# Top 10 Labels
sorted_dict = dict(sorted(cnt.items(), key=lambda item: item[1], reverse=True))
top10dict = dict(list(sorted_dict.items())[:10])

plt.figure(figsize=(10, 5))
plt.bar(top10dict.keys(), top10dict.values())
plt.xlabel('Labels')
plt.ylabel('Count')
plt.title('Top 10 Labels Distribution')
plt.xticks(rotation=45)
plt.show()

# Load Images
X = []
y = []
for filename in os.listdir(directory_path):
    if os.path.isfile(os.path.join(directory_path, filename)):
        name = filename.split("_")[0]
        if name in top10dict:
            X.append(cv2.imread(os.path.join(directory_path, filename)))
            y.append(name)

# Check Image Shapes
image_shapes = [img.shape for img in X]
unique_shapes = set(image_shapes)
print(f"Unique image shapes: {unique_shapes}")

# Assuming the images have different shapes, we'll need to resize them to a common size
# For simplicity, let's assume a common size of (224, 224)
X_resized = [cv2.resize(img, (224, 224)) for img in X]

# Calculate Image Statistics
X_array = np.array(X_resized)
mean = np.mean(X_array, axis=(0, 1, 2))
std = np.std(X_array, axis=(0, 1, 2))
print(f"Image Mean: {mean}")
print(f"Image Std Dev: {std}")

# Plot pixel value distributions for each channel
plt.figure(figsize=(12, 6))
colors = ['r', 'g', 'b']
for i, color in enumerate(colors):
    plt.hist(X_array[..., i].ravel(), bins=256, color=color, alpha=0.5, label=f'{color} channel')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.title('Pixel Intensity Distribution')
plt.legend()
plt.show()

# Display a grid of sample images
fig, axs = plt.subplots(2, 5, figsize=(15, 6))
axs = axs.flatten()
for img, ax, label in zip(X_resized[:10], axs, y[:10]):
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.set_title(f"Label: {label}")
    ax.axis('off')
plt.tight_layout()
plt.show()


# Label Count Distribution
plt.figure(figsize=(14, 7))
all_labels = list(cnt.keys())
all_counts = list(cnt.values())
sns.barplot(x=all_labels, y=all_counts)
plt.xlabel('Labels')
plt.ylabel('Count')
plt.title('Label Count Distribution')
plt.xticks(rotation=90)
plt.show()

# Image Aspect Ratios
aspect_ratios = [img.shape[1] / img.shape[0] for img in X]
plt.figure(figsize=(10, 5))
plt.hist(aspect_ratios, bins=30)
plt.xlabel('Aspect Ratio (Width/Height)')
plt.ylabel('Frequency')
plt.title('Distribution of Image Aspect Ratios')
plt.show()


random_indices = random.sample(range(len(X_resized)), 10)
fig, axs = plt.subplots(2, 5, figsize=(15, 6))
axs = axs.flatten()
for i, ax in zip(random_indices, axs):
    ax.imshow(cv2.cvtColor(X_resized[i], cv2.COLOR_BGR2RGB))
    ax.set_title(f"Label: {y[i]}")
    ax.axis('off')
plt.tight_layout()
plt.show()

# Class-wise Mean Image
class_mean_images = {}
for label in top10dict.keys():
    class_images = [X_resized[i] for i in range(len(y)) if y[i] == label]
    mean_image = np.mean(class_images, axis=0).astype(np.uint8)
    class_mean_images[label] = mean_image

fig, axs = plt.subplots(2, 5, figsize=(15, 6))
axs = axs.flatten()
for ax, (label, mean_image) in zip(axs, class_mean_images.items()):
    ax.imshow(cv2.cvtColor(mean_image, cv2.COLOR_BGR2RGB))
    ax.set_title(f"Mean Image: {label}")
    ax.axis('off')
plt.tight_layout()
plt.show()

# Image Size Distribution
widths = [img.shape[1] for img in X]
heights = [img.shape[0] for img in X]

plt.figure(figsize=(10, 5))
plt.hist(widths, bins=30, alpha=0.5, label='Widths')
plt.hist(heights, bins=30, alpha=0.5, label='Heights')
plt.xlabel('Size (pixels)')
plt.ylabel('Frequency')
plt.title('Distribution of Image Sizes')
plt.legend()
plt.show()

# Class-wise Standard Deviation Image
class_std_images = {}
for label in top10dict.keys():
    class_images = [X_resized[i] for i in range(len(y)) if y[i] == label]
    std_image = np.std(class_images, axis=0).astype(np.uint8)
    class_std_images[label] = std_image

fig, axs = plt.subplots(2, 5, figsize=(15, 6))
axs = axs.flatten()
for ax, (label, std_image) in zip(axs, class_std_images.items()):
    ax.imshow(cv2.cvtColor(std_image, cv2.COLOR_BGR2RGB))
    ax.set_title(f"Std Dev Image: {label}")
    ax.axis('off')
plt.tight_layout()
plt.show()


# Flatten the images
X_flattened = np.array([img.flatten() for img in X_resized])

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_flattened)

# Plot PCA results
plt.figure(figsize=(10, 7))
for label in top10dict.keys():
    indices = [i for i in range(len(y)) if y[i] == label]
    plt.scatter(X_pca[indices, 0], X_pca[indices, 1], label=label, alpha=0.6)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('PCA of Image Data')
plt.legend()
plt.show()

def tsne():
    # Apply T-SNE
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X_flattened)

    # Plot T-SNE results
    plt.figure(figsize=(10, 7))
    for label in top10dict.keys():
        indices = [i for i in range(len(y)) if y[i] == label]
        plt.scatter(X_tsne[indices, 0], X_tsne[indices, 1], label=label, alpha=0.6)
    plt.xlabel('T-SNE Component 1')
    plt.ylabel('T-SNE Component 2')
    plt.title('T-SNE of Image Data')
    plt.legend()
    plt.show()

# tsne()


# Plot class-wise pixel intensity distributions
plt.figure(figsize=(15, 10))
for i, label in enumerate(top10dict.keys()):
    plt.subplot(2, 5, i+1)
    class_images = [X_resized[j] for j in range(len(y)) if y[j] == label]
    class_pixel_values = np.array(class_images).ravel()
    plt.hist(class_pixel_values, bins=256, color='gray', alpha=0.7)
    plt.title(f'{label}')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
plt.tight_layout()
plt.show()



# Apply edge detection
edges = [cv2.Canny(img, 100, 200) for img in X_resized]

# Display edge-detected images
fig, axs = plt.subplots(2, 5, figsize=(15, 6))
axs = axs.flatten()
for i, ax in zip(range(10), axs):
    ax.imshow(edges[i], cmap='gray')
    ax.set_title(f"Label: {y[i]}")
    ax.axis('off')
plt.tight_layout()
plt.show()



# Calculate brightness and contrast
brightness = [np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)) for img in X_resized]
contrast = [np.std(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)) for img in X_resized]

# Plot brightness distribution
plt.figure(figsize=(10, 5))
plt.hist(brightness, bins=30, color='blue', alpha=0.7)
plt.xlabel('Brightness')
plt.ylabel('Frequency')
plt.title('Distribution of Image Brightness')
plt.show()

# Plot contrast distribution
plt.figure(figsize=(10, 5))
plt.hist(contrast, bins=30, color='red', alpha=0.7)
plt.xlabel('Contrast')
plt.ylabel('Frequency')
plt.title('Distribution of Image Contrast')
plt.show()



# Downsample images to reduce dimensions
X_downsampled = [cv2.resize(img, (56, 56)) for img in X_resized]  # Example: Downsampling to 56x56

# Flatten the downsampled images
X_flattened_downsampled = np.array([img.flatten() for img in X_downsampled])

# Normalize pixel values
X_flattened_downsampled_normalized = X_flattened_downsampled / 255.0

# Compute the correlation matrix
correlation_matrix_downsampled = np.corrcoef(X_flattened_downsampled_normalized, rowvar=False)

# Plot correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix_downsampled, cmap='viridis')
plt.title('Correlation Matrix of Downsampled Pixel Values')
plt.show()
