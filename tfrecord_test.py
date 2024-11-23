import tensorflow as tf

# Path ke file TFRecord
tfrecord_file = "data/tf_records/test-no-dup-00000-of-00005"

# Fungsi untuk membaca TFRecord
def read_tfrecord(file_path):
    try:
        raw_dataset = tf.data.TFRecordDataset(file_path)
        for raw_record in raw_dataset.take(1):  # Membaca 1 record sebagai contoh
            print("TFRecord ditemukan!")
            print("Isi record pertama (biner):", raw_record.numpy())
            return True
    except Exception as e:
        print("Bukan file TFRecord atau file corrupt.")
        print("Error:", str(e))
        return False

# Periksa file
is_tfrecord = read_tfrecord(tfrecord_file)
if is_tfrecord:
    print(f"{tfrecord_file} adalah file TFRecord.")
else:
    print(f"{tfrecord_file} bukan file TFRecord.")
