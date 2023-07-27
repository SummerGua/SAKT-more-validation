from SAKT import SAKTModel
from prepare_data import SAKTDataset
from utils import values_after_mask
import torch
from torch.utils.data import DataLoader
import pandas as pd

n_skill = 100
max_len = 50
emb_dim = 100
num_heads = 5
dropout = 0.2
seed = 3407

device = torch.device('cpu')
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

sakt = SAKTModel(n_skill, emb_dim, num_heads, max_len, device, dropout).to(device)
sakt.load_state_dict(torch.load("model_params.pkl"))

test_data_path = "./data/mock_test.csv"
test_df = pd.read_csv(test_data_path, header=None, sep='\t')
test_data = SAKTDataset(test_df, n_skill, max_len)
test_dataloader = DataLoader(test_data, shuffle=False)

for i, (qa, qid, labels, mask) in enumerate(test_dataloader):
    qa, qid, labels, mask = (
        qa.to(device),
        qid.to(device),
        labels.to(device),
        mask.to(device),
    )

    with torch.no_grad():
        pred = sakt(qid, qa)
    real_pred, _ = values_after_mask(pred, labels, mask)
    print(real_pred)
