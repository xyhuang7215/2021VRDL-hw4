## VRDL HW4

SRResNet for super-resolution 

[Reference](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Super-Resolution)

## Colab
You may use [Colab notebook](https://colab.research.google.com/drive/1MOqrqvGXgX_5dL4YEUCYepydTzmmBT0J?usp=sharing)(recommend) to get the inference result, or follow the instruction below.

## Requirements
Install the enviroment and get the dataset/checkpoint.

```train
pip install -r requirements.txt

pip install gdown==4.2.0
gdown https://drive.google.com/uc?id=1eLw2aCziPdR8qxcH9xCMQLxzsWkenyO9
gdown https://drive.google.com/uc?id=141nULhR0wdqWO0vRdHKo8AXMY4h49BQe
unzip datasets.zip
```

## Training

```train
python train.py
```

## Inference

```eval
python inference.py 
```
The inference images will be saved in the test folder.
