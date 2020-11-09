from customizedDataset import customizedDataset

import torch

if __name__ == "__main__":
    print('currently executing main.py file')
    
    trainLoader = torch.utils.data.DataLoader(customizedDataset(),batch_size = 1, shuffle = True, num_workers = 0)

    for idx, data in enumerate(trainLoader):
        print(idx)