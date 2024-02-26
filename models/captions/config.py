import torch 


csv_file = "coco/coco2017/annotations/captions_train2017.json"
images_dir = "coco/coco2017/train2017"

val_csv_file = "coco/coco2017/annotations/captions_val2017.json"
val_images_dir = "coco/coco2017/val2017"

train_dataset_ratio = 0.9
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    torch.cuda.empty_cache()
max_steps = 100
batch_size = 100
shuffle = True
num_workers = 4
max_len = 52
dropout = 0.1
glove_file = '.vector_cache/glove.6B.300d.txt'