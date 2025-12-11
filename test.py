import face_recognition
import numpy as np 
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image
import torch
import random

#ML Helper Libraries
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE
from hdbscan import HDBSCAN


#libraries for file manipulation
import glob as glob 
import csv, os

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#Silencing warnings — these aren't hurting performance
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# #pytorch params
device = 'cpu'

mtcnn = MTCNN(
    image_size=182,
    margin=40,
    keep_all=False,
    post_process=True,
    device=device
)

resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

"""Step 1: Embed images to 512d vectors -> cluster images by identity using DBSCAN"""

#load the images
images = glob.glob("faces/*.jpg")

#extract embeddings
embeddings = [] 
valid_paths = [] #as I sift through the images, only added valid ones for clustering
print("Encoding Images... ")

#used for face_recognition and DBSCAN
# for img in images:
#     image = face_recognition.load_image_file(img)
#     encode = face_recognition.face_encodings(image)

    
#     if len(encode) > 0: #only append one face, i noticed there are other humans in the background. This might pose an issue. 
#         #print("One image vector length: ", len(encode[0]), "\nWhole encoding: ", len(encode)) #there might be a none
        
#         embeddings.append(encode[0])
#         valid_paths.append(img)
#     else:
#         print("Skipping:", img)

for img_path in images:
    img = Image.open(img_path).convert("RGB")

    # detect + align face
    face = mtcnn(img)

    if face is None:
        print("No face detected:", img_path)
        continue

    face = face.unsqueeze(0).to(device)

    with torch.no_grad():
        # 512-D embedding
        embedding = resnet(face).cpu().numpy()[0]

    embeddings.append(embedding)
    valid_paths.append(img_path)

    

#print(len(embeddings)) #its 405, so good

#-----clustering images by identity ---

#used for face_recognition model and DBSCAN
# X = np.array(embeddings, dtype=float)
# X = normalize(X, norm="l2")
# print(X) #successful numpy array, there were two images in some. So it is detecting the background. Easy fix
# print(X[0].shape)

# #drop bad vectors if any
# mask = np.isfinite(X).all(axis=1)
# X = X[mask]
# valid_paths = [p for m,p in zip(mask, valid_paths) if m]

# print("Final embedding count:", X.shape)
# print("Embedding norms:", np.linalg.norm(X, axis=1).min(), np.linalg.norm(X, axis=1).max())

# dbscan = DBSCAN(eps=0.35, min_samples=3, metric='euclidean') #watched a quick video on how dbscan works and found sklearn built in class and method
# labels = dbscan.fit_predict(X) # this collects the clusters and assigns them numbers all in one step. 

#print(labels)


# used for facenet-pytorch and HDBSCAN
X = np.array(embeddings, dtype=np.float32)

norms = np.linalg.norm(X, axis=1)
mask = norms > 1e-6
X = X[mask]
valid_paths = [p for m,p in zip(mask, valid_paths) if m]

X = normalize(X, norm="l2")

mask = np.isfinite(X).all(axis=1)
X = X[mask]
valid_paths = [p for m,p in zip(mask, valid_paths) if m]

print("After filtering:", X.shape)
print("NaNs:", np.isnan(X).sum(), "Infs:", np.isinf(X).sum())
print("Norm range:", np.linalg.norm(X, axis=1).min(), np.linalg.norm(X, axis=1).max())

#saving normalized embeddings
# np.savez("normalized_embeddings.npz",
#          embeddings=X.astype(np.float32),
#          paths=np.array(valid_paths))



# # ---- clustering ----
clusterer = HDBSCAN(
        min_cluster_size=3,
        min_samples=1,
        metric='euclidean',
        cluster_selection_epsilon=0.2)

labels = clusterer.fit_predict(X) 

# ---- HDBSCAN condensed tree diagnostic ----
# clusterer.condensed_tree_.plot(select_clusters=True)
# plt.title("HDBSCAN Condensed Tree")
# plt.show()

"""Step 2. Place the appropriate clusters into the cluster.csv as well as outliers.csv. Also, below is PCA of the 3D projection colored by cluster """
os.makedirs("artifacts", exist_ok=True)

#clusters.csv
with open("artifacts/clusters.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image_filename, cluster_id"])

    for label,path in zip(labels, valid_paths):
        writer.writerow([os.path.basename(path), label])

#outliers.csv
with open("artifacts/outliers.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image_filename, cluster_id"])

    for label,path in zip(labels, valid_paths):
        if label == -1:
            writer.writerow([os.path.basename(path), label])

for label, path in zip(labels, valid_paths):
    folder = f"artifacts/cluster_{label}"
    os.makedirs(folder, exist_ok=True)
    os.system(f"cp '{path}' '{folder}'")

#PCA for all clusters

cluster_amt = max(labels) #23 clusters
print(cluster_amt)

pca = PCA(n_components=10)
points3d = pca.fit_transform(X)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

scatter = ax.scatter(
    points3d[:,0],
    points3d[:,1],
    points3d[:,2],
    c=labels,
    s=15
)

explained_variance = pca.explained_variance_ratio_
print(f"Explained variance from the PCA: {explained_variance} \n Sum of the explained variance from the PCA: {sum(explained_variance)}")

#plt.savefig("artifacts/viz_3d.png", dpi=200)
# plt.close()

# t-SNE from the 512-dim vectors
tsne = TSNE(
    n_components=3,
    learning_rate='auto',
    init='random',
    perplexity=30
)

X_3d = tsne.fit_transform(X)
# Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(
    X_3d[:, 0],
    X_3d[:, 1],
    X_3d[:, 2],
    s=5
)

plt.title("t-SNE Projection of Face Embeddings")
plt.show()


#pick out random cluster and display images in that cluster

unique_labels = list(set(labels))
random_label = random.choice(unique_labels)
print(f"Selected Cluster: {random_label}")

#Get all the images in that cluster
cluster_images = [path for label, path in zip(labels, valid_paths) if label == random_label]
print(f"Number of images in cluster {random_label}: {len(cluster_images)}")

# Display the images in a grid
num_images = len(cluster_images)
cols = min(5, num_images)  # max 5 columns
rows = (num_images + cols - 1) // cols  # ceiling division

fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
if num_images == 1:
    axes = [axes]
else:
    axes = axes.flatten() if num_images > 1 else [axes]

for idx, img_path in enumerate(cluster_images):
    img = Image.open(img_path)
    axes[idx].imshow(img)
    axes[idx].axis('off')
    axes[idx].set_title(os.path.basename(img_path), fontsize=8)

# Hide any unused subplots
for idx in range(num_images, len(axes)):
    axes[idx].axis('off')

plt.suptitle(f'Cluster {random_label} - {num_images} images', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()





