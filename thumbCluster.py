from PIL import Image
import math
import glob
import os

def make_cluster_thumbnail(cluster_folder, thumb_size=(80,80), max_images=16):
    # collect images
    jpgs = glob.glob(os.path.join(cluster_folder, "*.jpg"))
    if len(jpgs) == 0:
        return
    
    # take first max_images
    imgs = jpgs[:max_images]
    
    # compute grid size
    n = len(imgs)
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    
    # create blank grid canvas
    grid_w = cols * thumb_size[0]
    grid_h = rows * thumb_size[1]
    grid = Image.new('RGB', (grid_w, grid_h), color=(255,255,255))
    
    # fill grid
    for idx, img_path in enumerate(imgs):
        img = Image.open(img_path).convert("RGB")
        img = img.resize(thumb_size)
        x = (idx % cols) * thumb_size[0]
        y = (idx // cols) * thumb_size[1]
        grid.paste(img, (x, y))
    
    # save
    out_path = os.path.join(cluster_folder, "thumbnail.png")
    grid.save(out_path)
    print("Saved:", out_path)


if __name__ == "__main__":

    artifact_root = "artifacts"
    cluster_dirs = [d for d in os.listdir(artifact_root) if d.startswith("cluster_")]

    for d in cluster_dirs:
        folder = os.path.join(artifact_root, d)
        make_cluster_thumbnail(folder)
