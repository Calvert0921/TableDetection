# TableDetection

## File tree can be represented as:
```bash
coursework2_groupO/
├── data/
│   ├── CW2-Dataset/
│   ├── RealSense/
│   └── spilit.py           # 
├── src/
│   ├── pipelineA/   
│   │   ├── 
│   │   └──      
│   ├── pipelineB/ 
│   │   ├── dataloader_depthEst.py          # Dataloader for depth estimator
│   │   ├── dataloader_tableClassifier.py   # Dataloader for table classifier
│   │   ├── test_depthEst.py                # Testing script for depth estimator, and also generate depth predictions for later training
│   │   ├── test_pipeline.py                # Testing script for whole pipeline B
│   │   ├── train_depthEst.py               # Training script for depth estimator
│   │   ├── train_tableClassifier.py        # Training script for table classifier
│   │   └── utils.py                        # Helper functions that can convert 16-bit depth to 8-bit
│   ├── pipelineC/  
│   │   ├── 
│   │   └──        
├── datasets/               # 
├── weights/                # Model weights
├── predictions/            # Predictions, plots
├── read_labels.py          # Script to read the label
├── requirements.txt        # Conda / pip environment specs
└── README.md               # Setup / usage instructions
```

## To install all the dependencies:
### 1. Run following command to create new environment
```bash
conda create --name object python=3.10
```

### 2. Install the suitable pytorch version

### 3. Run following script to install all necessary libraries
```bash
pip3 install -r requirements.txt
```

### 4. Prepare the data via the zip file

## Instruction for Pipeline A:

## Instruction for Pipeline B:

### 1. To train the depth estimator:
```bash
cd src/pipelineB
python3 train_depthEst.py
```

### 2. To test the depth estimator and also get the training samples for table classifier(Adjust the test path in line 106 if needed):
```bash
python3 test_depthEst.py
```

### 3. To train the table classifier using combined data:
```bash
python3 train_tableClassifier.py
```

### 4. To test the whole pipeline(Adjust the test path in line 126 if needed):
```bash
python3 test_pipeline.py
```
### **The depth prediction will be saved in "depthPred" under each data folder and all other results will be printed in terminal in the form of "Classification Report".**

## Instruction for Pipeline C:

