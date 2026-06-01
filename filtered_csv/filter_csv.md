# About Filter_csv

## 文件夹描述

该文件夹为实验所用数据，数据来源为random split（分层）

目录下的四个文件为：
* Test_sgf.csv :sgf任务所使用的测试数据集
* Train_sgf.csv ：sgf任务所使用的训练数据集
* Test_sif.csv ：sif任务所使用的测试数据集
* Train_sif.csv ： sif任务所使用的训练数据集

目录下的三个文件夹为，对应专利名称的所有数据：
sif_sgf_second/ 
US9624268/
WO2017011820A2/
其中，每个专利文件夹下包含两个csv文件，对应专利中可用于SIF和SGF两个任务的数据集
*/SIF.csv
*/SGF.csv

## 数据描述
每一个csv的数据都包含以下列
id：            数据id，但是没有用，不要参考这个
is_monomer：    是否为单分子，默认全是True
SMILES          分子的字符串表示
SIF_minutes     在SIF任务下的半衰期
SGF_minutes     在SGF任务下的半衰期
source_name     数据来源

## 专利描述
目前一共有五个专利，专利包括如下五个
【特别注意，专利名中包括csv，代表最原始的来源，这五个文件实际并不存在于当前项目】
```
SOURCES = [
    "sif_sgf_second.csv",
    "US20140294902A1.csv",
    "US9624268.csv",
    "US9809623B2.csv",
    "WO2017011820A2.csv",
]
```


