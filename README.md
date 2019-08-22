# Learning Discrete and Continuous Factors of Data via Alternating Disentanglement
## Demo
[![](http://img.youtube.com/vi/pRsD0Ot26gw/0.jpg)](http://www.youtube.com/watch?v=pRsD0Ot26gw "Learning Discrete and Continuous Factors of Data via Alternating Disentanglement")

## Dependency
- python=3.5
- tensorflow version = 1.4
- CUDA 8.0
- cuDNN 6.0
- Environment detail is listed in `ex.yml'

## Citing this work
```
@inproceedings{jeongICML19,
    title={
        Learning Discrete and Continuous Factors of Data via Alternating Disentanglement
    },
    author= {Yeonwoo Jeong and Hyun Oh Song},
    booktitle={International Conference on Machine Learning (ICML)},
    year={2019}
}
```

## Dataset(dSprites)
- Download from [https://github.com/deepmind/dsprites-dataset](https://github.com/deepmind/dsprites-dataset)

## Edit path
- Edit path in 'config/path.py'
- ROOT - (directory for experiment result)
- DSPRITESPATH - (directory for downloaed dsprites)

## Run model
- Dsprites_exp/CascadeVAE/main.py
- Dsprites_exp/CascadeVAE-C/main.py

## Trained model
 - Download from [here](https://drive.google.com/file/d/1GTP2uUCJVaU3nXG1Tk2G-BFiTUlCM5k2/view?usp=sharing).
 - Here are trained models from 10 different random seeds. 

## License
MIT License 
