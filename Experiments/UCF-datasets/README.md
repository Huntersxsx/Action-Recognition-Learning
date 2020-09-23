[UCF-101官网](https://www.crcv.ucf.edu/data/UCF101.php/)下载数据集后解压  
结构如下：

UCF-101
├── ApplyEyeMakeup
│   ├── v_ApplyEyeMakeup_g01_c01.avi
│   └── ...
├── ApplyLipstick
│   ├── v_ApplyLipstick_g01_c01.avi
│   └── ...
└── Archery
│   ├── v_Archery_g01_c01.avi
│   └── ...


运行dataset.py，生成训练集、测试集、验证集
结构如下：

preprocessed
├── train
	├── ApplyEyeMakeup
	│   ├── v_ApplyEyeMakeup_g01_c01
	│   │   ├── 00001.jpg
	│   │   └── ...
	│   └── ...
	├── ApplyLipstick
	│   ├── v_ApplyLipstick_g01_c01
	│   │   ├── 00001.jpg
	│   │   └── ...
	│   └── ...
	└── Archery
	│   ├── v_Archery_g01_c01
	│   │   ├── 00001.jpg
	│   │   └── ...
	│   └── ...
├── val

├── test