'''
prepare_data.py data format transformation
current format：
1,2,3,4  0,1,1,0 
skill_id answer

target format：
skill  =  [1,   2,   3,  4,      n_skill,....,n_skill]
qa     =  [x,   x,   x,  x,      n_skill*2+1, ....,  ] # x = skill_id + answer * n_skill
labels =  [0,   1,   1,  0,      -1,-1,-1,-1, ..., -1] # we need to predict "0" in index 3
mask   =  [0, 0.9, 0.9,  1,      0,0,0,0,0,0, ...., 0] # >0 are used to calculate loss. =1 is used to calculate auc in test set
'''
import torch.utils.data
import torch.nn.utils
import numpy as np

class SAKTDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, n_skill, max_len=200):
        super(SAKTDataset, self).__init__()
        self.df = dataframe
        self.n_skill = n_skill
        self.max_len = max_len

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        qids = self.df[0][index].split(",") # in fact skill_id
        correct = self.df[1][index].split(",")

        # remain the last max_len interactions
        if len(qids) > self.max_len:
            qids = qids[-self.max_len:]
            correct = correct[-self.max_len:]

        # int -> list -> ndarray
        qids = np.array(list(map(int, qids)))
        correct = np.array(list(map(int, correct)))

        skill = np.ones(self.max_len) * self.n_skill
        skill[:len(qids)] = qids


        mask = np.zeros(self.max_len)
        mask[1:len(correct)-1] = 0.9 # the first pred is invalid, because it has no historical info
        mask[len(correct)-1] = 1 # the last one is waht we want to predict: P(a_t | q_t, qa_{<t})

        labels = np.ones(self.max_len) * -1
        labels[:len(correct)] = correct

        qa = np.ones(self.max_len) * (self.n_skill * 2 + 1)
        qa[:len(qids)] = qids + correct * self.n_skill
        qa[len(qids)-1] = self.n_skill * 2 + 1
        return (
            torch.cat(
                (torch.LongTensor([2 * self.n_skill]), torch.LongTensor(qa[:-1]))
            ), #qa
            torch.LongTensor(skill),
            torch.LongTensor(labels), # truth
            torch.FloatTensor(mask),
        )