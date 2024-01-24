# EKRD
This is the Pytorch implementation for our paper: **Efficient Knowledge-aware Recommendation via Distillation**.

# Enviroment Requirement
* torch=1.9.0
* python=3.7.15
* scipy=1.7.3
* numpy=1.21.6
* torch-scatter=2.0.9

# Dataset
We provide three processed datasets and the corresponding knowledge graphs: MovieLens and LastFM and Amazon-book.

# How to Run the Code
You need to create the `History/` directory. The command to train EKRD on the CKE teacher model on  MovieLens/LastFM/Amazon-book dataset is as follows.
* MovieLens
  
```python main.py --lsreg=3e-4 --aglreg=10 --reg=3e-9 --cdreg=1e-3 --softreg=1e-3 --dataset=movie-lens --teacher_model=CKE```

* LastFM

```python main.py --lsreg=1e-3 --aglreg=3 --reg=3e-10 --cdreg=1e-3 --softreg=3e-3 --dataset=last-fm --teacher_model=CKE```

* Amazon-book

```python main.py --lsreg=3e-3 --aglreg=10 --reg=3e-9 --cdreg=3e-3 --softreg=1e-4 --dataset=amazon-book --teacher_model=CKE```
