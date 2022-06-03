![](https://img.shields.io/badge/LSTM-Load%20Forecasting-red)
# LSTM-Load-Forecasting
Implementation of Electric Load Forecasting Based on LSTM(BiLSTM). Including Univariate-SingleStep forecasting, Multivariate-SingleStep forecasting and Multivariate-MultiStep forecasting.

# Environment
pytorch==1.10.1+cu111

numpy==1.18.5

pandas==1.2.3

# Tree
```bash
.
│  args.py
│  data_process.py
│  LICENSE
│  models.py
│  README.md
│  tree.txt
│  util.py
│          
├─data
│      data.csv
│      
├─LSTMs
│      multivariate_multi_step.py
│      multivariate_single_step.py
│      univariate_single_step.py
│      
└─models
        multivariate_multi_step.pkl
        multivariate_single_step.pkl
        univariate_single_step.pkl
```
1. **args.py** is a parameter configuration file, where you can set model parameters and training parameters.
2. **data_process.py** is the data processing file. If you need to use your own data, then you can modify the load_data function in data_process.py.
3. Two models are defined in **models.py**, including LSTM and bidirectional LSTM.
4. **util.py** defines the training and testing functions of the models in the three prediction methods.
5. The trained model is saved in the **models** folder, which can be used directly for testing.
6. Data files in csv format are saved under the **data** file.
# Usage
First switch the working path:
```bash
cd LSTMs/
```
Then, execute in sequence:
```bash
python multivariate_multi_step.py --epochs 50 --batch_size 30
python multivariate_single_step.py --epochs 30 --batch_size 30
python univariate_single_step.py --epochs 30 --batch_size 30
```
If you need to change the parameters, please modify them manually in args.py.
# Result
![在这里插入图片描述](https://img-blog.csdnimg.cn/2afb0a892c854ca39a46263b25b57d5a.png#pic_center)
