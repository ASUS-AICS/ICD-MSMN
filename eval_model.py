from main import eval_func
import sys
import os
import torch
import json
from torch.utils.data import DataLoader
from data_util import MimicFullDataset, my_collate_fn
from evaluation import all_metrics, print_metrics
from pandas import DataFrame
import numpy as np
import copy
from find_threshold import find_threshold_micro


if __name__ == "__main__":
    device = "cuda:0"
    model_path = sys.argv[1]
    version = sys.argv[2]
    
    print(f"Version: {version}")

    model = torch.load(model_path).to(device)

    word_embedding_path = sys.argv[3]
    length = int(sys.argv[4])
    dev_dataset = MimicFullDataset(version, "dev", word_embedding_path, length)
    dev_dataloader = DataLoader(dev_dataset, batch_size=4, collate_fn=my_collate_fn, shuffle=False, num_workers=1)
    test_dataset = MimicFullDataset(version, "test", word_embedding_path, length)
    test_dataloader = DataLoader(test_dataset, batch_size=4, collate_fn=my_collate_fn, shuffle=False, num_workers=1)
    dev_metric, (dev_yhat, dev_y, dev_yhat_raw), threshold = eval_func(model, dev_dataloader, device, tqdm_bar=True)
    print('Default Threshold on Dev')
    print_metrics(dev_metric, suffix="Dev")

    if isinstance(threshold, float) or (version == 'mimic3-50'):
        print('Threshold:', threshold)

    print(f'Adjust Threshold on Test')
    test_metric, (test_yhat, test_y, test_yhat_raw), _ = eval_func(model, test_dataloader, device, tqdm_bar=True, threshold=threshold)
    
    np.save(model_path.replace('.pth', '-gts.npy'), test_y)
    np.save(model_path.replace('.pth', '-preds.npy'), test_yhat_raw)
    
    with open(model_path.replace('.pth', '-label-dict.json'), 'w+') as f:
        json.dump([str(v) for _, v in test_dataset.ind2c.items()], f)

    print_metrics(test_metric, suffix='Test')
