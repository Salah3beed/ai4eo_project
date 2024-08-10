import fiftyone as fo
name = "SARD_YOLOv5"

train_dir = "SARD_YOLOv5/train"
valid_dir = "SARD_YOLOv5/valid"
test_dir = "SARD_YOLOv5/test"

# names must be different for each dataset
train_dataset = fo.Dataset.from_dir(
    dataset_dir=train_dir,
    dataset_type=fo.types.YOLOv5Dataset,
    name=f"{name}-train",
)

valid_dataset = fo.Dataset.from_dir(
    dataset_dir=valid_dir,
    dataset_type=fo.types.YOLOv5Dataset,
    name=f"{name}-valid",
)

test_dataset = fo.Dataset.from_dir(
    dataset_dir=test_dir,
    dataset_type=fo.types.YOLOv5Dataset,
    name=f"{name}-test",
)


# Export directories for each split
train_export_dir = "YOLOv5_VOC/train"
valid_export_dir = "YOLOv5_VOC/valid"
test_export_dir = "YOLOv5_VOC/test"

train_dataset.export(
    export_dir=train_export_dir,
    dataset_type=fo.types.VOCDetectionDataset,
    label_field="ground_truth",  
)

valid_dataset.export(
    export_dir=valid_export_dir,
    dataset_type=fo.types.VOCDetectionDataset,
    label_field="ground_truth", 
)

test_dataset.export(
    export_dir=test_export_dir,
    dataset_type=fo.types.VOCDetectionDataset,
    label_field="ground_truth",  
)