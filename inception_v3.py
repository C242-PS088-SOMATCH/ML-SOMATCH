import requests
import tarfile
import os

# URL model Inception v3
url = "https://storage.googleapis.com/download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz"
save_dir = "checkpoints/inception_v3/"
tar_file = os.path.join(save_dir, "inception_v3_2016_08_28.tar.gz")

# Membuat folder jika belum ada
os.makedirs(save_dir, exist_ok=True)

# Mengunduh file
print("Mengunduh Inception v3 checkpoint...")
response = requests.get(url, stream=True)
with open(tar_file, "wb") as f:
    for chunk in response.iter_content(chunk_size=1024):
        if chunk:
            f.write(chunk)

print("Ekstrak file checkpoint...")
with tarfile.open(tar_file, "r:gz") as tar:
    tar.extractall(path=save_dir)

print("Selesai! Checkpoint tersedia di:", save_dir)
