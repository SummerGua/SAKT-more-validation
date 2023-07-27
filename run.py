from prepare_data import SAKTDataset
from SAKT import SAKTModel
from SAKTLoss import SAKTLoss
from utils import train_one_epoch, eval_one_epoch
import argparse
import torch
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm

seed = 3407

def run(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print("training on " + args.dataset_name + "...")

    # load data
    train_data_path = "./data/" + args.dataset_name + "_train.csv"
    test_data_path = "./data/" + args.dataset_name + "_test.csv"
    train_df = pd.read_csv(train_data_path, header=None, sep='\t')
    test_df = pd.read_csv(test_data_path, header=None, sep='\t')

    train_data = SAKTDataset(train_df, args.n_skill, args.max_len)
    test_data = SAKTDataset(test_df, args.n_skill, args.max_len)

    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size*2, shuffle=False)

    sakt = SAKTModel(args.n_skill, args.embed_dim, args.num_heads, args.max_len, device, args.dropout).to(device)
    # load model params
    # sakt.load_state_dict(torch.load("model_params.pkl"))
    optimizer = torch.optim.Adam(sakt.parameters(), lr=args.learning_rate)
    sakt_loss = SAKTLoss().to(device)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    for epoch in tqdm(range(args.epoch)):
        train_one_epoch(sakt, train_dataloader, optimizer, sakt_loss, args.save_params,device)
        eval_one_epoch(sakt, test_dataloader, device)
        scheduler.step()


if __name__ == "__main__":
    
    arg_parser = argparse.ArgumentParser(description="train SAKT")
    arg_parser.add_argument("--learning_rate",
                            dest="learning_rate",
                            default=0.001,
                            type=float,
                            required=False)
    arg_parser.add_argument("--batch_size",
                            dest="batch_size",
                            default=128,
                            type=int,
                            required=False)
    arg_parser.add_argument("--n_skill",
                            dest="n_skill",
                            default=100, # ASSIST2015 = 100
                            type=int,
                            required=False)
    arg_parser.add_argument("--embed_dim",
                            dest="embed_dim",
                            default=100,
                            type=int,
                            required=False)
    arg_parser.add_argument("--dropout",
                            dest="dropout",
                            default=0.2,
                            type=float,
                            required=False)
    arg_parser.add_argument("--num_heads",
                            dest="num_heads",
                            default=5,
                            type=int,
                            required=False)
    arg_parser.add_argument("--epoch",
                            dest="epoch",
                            default=20, # 15
                            type=int,
                            required=False)
    arg_parser.add_argument("--num_worker",
                            dest="num_worker",
                            default=0,
                            type=int,
                            required=False)
    arg_parser.add_argument("--dataset_name",
                            dest="dataset_name",
                            # default="mock",
                            default="assist2015",
                            type=str,
                            required=False)
    arg_parser.add_argument("--max_len",
                            dest="max_len",
                            default=50,
                            type=int,
                            required=False)
    arg_parser.add_argument("--save_params",
                            dest="save_params",
                            default=False,
                            type=bool,
                            required=False)
    args = arg_parser.parse_args()

    run(args)
