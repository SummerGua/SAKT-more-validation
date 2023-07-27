# SAKT Implemented in PyTorch

## Introduction
This project expands the implemention of SAKT model in [deepKT](https://github.com/jdxyw/deepKT) with **better loss calculation and various auc & acc results**.

Results show auc scores calculated with the last answer (y_t) are more closed to that in [SAKT paper](https://arxiv.org/abs/1907.06837). **We think that most students' last answer is `1` leads to very high auc and acc calculated with y_t.**

- train: define hyper-params and run `$python run.py`
- predict: choose your `.csv` file and run `$python predict.py`

![](./assets/auc_in_ten_epochs.png)
![](./assets/auc_in_paper.png)

## Method of This Project:

Say we have an interaction sequence (n_skill=100):
- concept id: 1,2,3,4
- student response: 1,1,1,1

![](./assets/draft.jpg)

**We also keep the method in deepKT. See the console.**

![](./assets/auc_no_first.png)

## Prediction

Say 5 students have finished 4 exercises corresponding to concept_25 with different answers.
```python
25,25,25,25,25	1,1,1,1,1 # all correct
25,25,25,25,25	0,1,1,1,1
25,25,25,25,25	0,0,1,1,1
25,25,25,25,25	0,0,0,1,1
25,25,25,25,25	0,0,0,0,1 # all wrong
```
Let's predict the 5th answer on concept_25. Prediction value should decrement?

![](./assets/pred.png)

Good. But sometimes fluctuate :( So we should try AKT / SAINT / SAINT+

![](./assets/fluctuate.png)

And different last answers(i.e. the ground truths) won't change the result, which is what we want.

![](./assets/pred_no_change.png)

## Reference:

- *Pandey, Shalini, and George Karypis. "A self-attentive model for knowledge tracing." arXiv preprint arXiv:1907.06837 (2019).* - main reference
- https://github.com/jdxyw/deepKT - main reference
- *Ghodai Abdelrahman, Qing Wang, and Bernardo Nunes. "Knowledge Tracing: A Survey". ACM Comput. Surv. 55, 11, Article 224 (2023), 37 pages.*
- https://github.com/arshadshk/SAKT-pytorch
- https://github.com/pykt-team/pykt-toolkit