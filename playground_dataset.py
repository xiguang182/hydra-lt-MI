import os
import csv
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import DatasetFolder

class CSVDIRDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.samples = self._make_dataset()
    
    def _make_dataset(self):
        data = []
        for record_idx, class_dir in enumerate(sorted(os.listdir(self.root))):
            class_path = os.path.join(self.root, class_dir)
            if os.path.isdir(class_path):
                for file_name in sorted(os.listdir(class_path)):
                    if file_name.endswith('.csv'):
                        file_path = os.path.join(class_path, file_name)
                        data.append(file_path)
        return data
    
    def __getitem__(self, index):
        path = self.samples[index]
        sample = self._load_csv(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample
    
    def __len__(self):
        return len(self.samples)
    
    def _load_csv(self, path):
        with open(path, 'r') as f:
            reader = csv.reader(f)
            line = next(reader)
            if line == '':
                raise ValueError('Empty file', path)
            return np.array(line, dtype=np.float32)



# hard coded dataset dir names
class CompositeMIDataset(Dataset):
    def __init__(self, root = './data', transform=None):
        self.root = root
        feature_dir = ['lang_data', 'OpenFace_Client_norm_dir', 'OpenFace_Counselor_norm_dir']
        self.labels_dir = 'Stable_mod_source_data_confirmed_20240213'
        self.datasets = [
            CSVDIRDataset(root = os.path.join(self.root, feature_dir[0]), transform=transform),
            CSVDIRDataset(root = os.path.join(self.root, feature_dir[1]), transform=transform),
            CSVDIRDataset(root = os.path.join(self.root, feature_dir[2]), transform=transform)
        ]
        self.labels, self.dividers = self.make_labels()
        assert len(self.datasets[0]) == len(self.datasets[1]) == len(self.datasets[2]) == len(self.labels), 'all child datasets and labels are from the same experiment data, and should be the same size'
        # the sequence length of the LSTM - 1, i.e. 6 length includes 1 of the target and 5 previous window
        self.window = 5


    def make_labels(self):
        labels = []
        dividers = [0]
        count = 0
        label_path = os.path.join(self.root, self.labels_dir)
        # labels should contains the same order of the other datasets
        for record_idx, file_name in enumerate(sorted(os.listdir(label_path))):
            if file_name.endswith('.csv'):
                file_path = os.path.join(label_path, file_name)
                with open(file_path, 'r', encoding= 'utf-8') as f:
                    reader = csv.reader(f)
                    for id, line in enumerate(reader):
                        # Skip the first line (header)
                        if id == 0:
                            continue
                        # Process each line (e.g., convert to float and print)
                        processed_line = []
                        for idx, item in enumerate(line):
                            # the first column (sp) speaker id
                            if idx == 0:
                                if item == 'Counselor':
                                    processed_line.append(1)
                                elif item == 'Client':
                                    processed_line.append(0)
                                else:
                                    raise ValueError('Unknown speaker id')
                            # the second column (cat) category, the actual label
                            if idx == 1:
                                processed_line.append(int(item))
                            # the third column (utt) utterance, skip
                            if idx == 2:
                                pass
                            # the fourth column (lang_file) RoBERTa file name 
                            if idx == 3:
                                processed_line.append(item)

                        # processed_line [speaker id, category, RoBERTa file name] 
                        labels.append(processed_line)
                        count += 1
                    dividers.append(count)
        assert dividers[-1] == len(labels)           
        return labels, dividers
    
    
    def index_translation(self, index, reverse=False):
        """
        Translate the index of the sequences data to the index of single data and vice versa
        param index: the index of the data
        param reverse: if True, translate the index of sequence data to the index of single data
        return: the translated index
        """
        # translate sequences data index to the index of single data
        if reverse:
            for divider in self.dividers:
                if index >= divider:
                    index += 5
        # translate single data index to the index of sequences data
        else:
            count = 0
            for divider in self.dividers:
                assert index >= (divider + 5), 'the first 5 data after every divider(file) should be skipped'
                count+=1
            index = index - count * 5
        return index
    
    def __getitem__(self, index):
        # Gather features from each dataset
        # translate the index of the sequences data to the index of single data
        index = self.index_translation(index, reverse=True)
        f1s = []
        f2s = []
        f3s = []
        for i in range(6):
            f1 = self.datasets[0][index-i]
            # append speaker id to the feature
            speaker_id = np.array(self.labels[index-i][0], dtype=np.float32)
            f1 = np.append(f1, speaker_id)
            f2 = self.datasets[1][index-i]
            f3 = self.datasets[2][index-i]
            f1s.append(f1)
            f2s.append(f2)
            f3s.append(f3)
        x1 = np.vstack(f1s)
        x2 = np.vstack(f2s)
        x3 = np.vstack(f3s)
        label = 0 if self.labels[index-i][1] > 33 else 1
        return (x1, x2, x3, label)
    
    def __len__(self):
        # all datasets are of equal length asserted in init
        # the total number of sequences data
        return len(self.datasets[0]) - self.window * len(self.dividers)



if __name__ == "__main__":
    # Usage
    root = './data/lang_data'
    dataset = CSVDIRDataset(root=root)

    # Create a DataLoader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)

    # Iterate through the DataLoader
    print(len(dataloader))
    for batch in dataloader:
        print(batch.shape, batch.dtype, batch)
        break

    d2 = CompositeMIDataset()
    dl2 = DataLoader(d2, batch_size=4, shuffle=False, num_workers=0)
    print(len(d2))
    print(d2[len(d2)-1])
    for idx, batch in enumerate(dl2):
        if idx % 100 == 0:
            print(idx)
        break