import os

labels_path = 'VisDrone_YOLO/VisDrone2019-DET-val/labels'
classes_to_ignore = [2,3,4,5,6,7,8,9]  

def update_labels(labels_path, classes_to_ignore):
    for label_file in os.listdir(labels_path):
        label_file_path = os.path.join(labels_path, label_file)
        if label_file_path.endswith('.txt'):
            with open(label_file_path, 'r') as file:
                lines = file.readlines()
            with open(label_file_path, 'w') as file:
                for line in lines:
                    parts = line.split()
                    class_idx = int(parts[0])
                    print(parts[:3])
                    if class_idx in classes_to_ignore:
                        continue  # skip the unwanted classes
                    if class_idx == 1:
                        parts[0] = '0'  # convert class 1 to class 0
                        print("OMG")
                    file.write(' '.join(parts) + '\n')

update_labels(labels_path, classes_to_ignore)