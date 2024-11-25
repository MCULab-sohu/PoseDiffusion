from torch.utils.data import Dataset
import numpy as np
import torch.nn.functional as F
import torch
def collate_fn(data):
    pixel_values = torch.stack([i['image'] for i in data])
    input_ids = torch.LongTensor([i['input_ids'] for i in data])
    return {'pixel_values': pixel_values, 'input_ids': input_ids}
class CustomDataset(Dataset):
    def __init__(self, data, tokenizer=None):
        self.data = data
        self.tokenizer = tokenizer
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data[idx]["image_path"]
        text = self.data[idx]["text"]
        
        image = np.load(image_path)
        image = F.interpolate(torch.from_numpy(image.reshape(17,64,64)).unsqueeze(0), size=(32, 32), mode='bilinear', align_corners=False).squeeze(0)

        tokenized_text = self.tokenizer(
                text,
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="pt",
            )
        tokenize = tokenized_text['input_ids'].tolist()[0]

        return {"image": image, "input_ids": tokenize}