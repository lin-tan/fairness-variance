# Calculating Confusion Matrix and Bias Metrics

## 1. Introduction
The code is a python project to calculate bias metrics of deep learning systems. It consists of two parts:

- Calculating confusion matrix from raw prediction results of deep learning system
- Calculating bias metrics based on confusion matrix



## 2. Code Structure
```bash
metric_calculation/
    |-- README.md
    |-- calculate.sh
    |-- calculate_ba.py
    |-- calculate_di.py
    |-- calculate_eo.py
    |-- calculate_fp.py
    |-- calculate_md.py
    |-- calculate_sp.py
    |-- confusion_matrix.py
    `-- utils.py
```

- `calculate.sh`: A bash script file to run metric calculation for all models
    - `calculate_ba.py`: A python file to calculate **Bias Amplification (BA)**
    - `calculate_di.py`: A python file to calculate **Disparate Impact (DI)**
    - `calculate_eo.py`: A python file to calculate **Equality of Opportunity (EO)**
    - `calculate_fp.py`: A python file to calculate **False Positive Subgroup Fairness (FPSF)**
    - `calculate_md.py`: A python file to calculate **Demographic Parity (DP)**
    - `calculate_sp.py`: A python file to calculate **Statistical Parity Subgroup Fairness (SPSF)**

- `confusion_matrix.py`: A python file to create confusion matrix for each classification result

- `utils.py`: A python file that contains utility functions used in metric calculation



## 3. Prerequisites

- Requirements
    - python >= 3.7.4
    - numpy  >= 1.20.2
    - pandas >= 0.25.1
    

- Prediction result of each model should in a CSV format. 
    - They should be placed on `metric_calculation/prediction/`. 

- Raw CSV file containing classification result should be formatted as:
    
    - Multi-class classification model 
        
    | idx 	| ground_truth 	| prediction_result 	| protected_label 	|
    |:---:	|:------------:	|:-----------------:	|:---------------:	|
    |  0  	|       3      	|         2         	|        0        	|
    |  1  	|       4      	|         4         	|        1        	|
    |  2  	|       3      	|         2         	|        0        	|

    - Multi-label classification model 

    | idx 	| ground_truth 	| prediction_result 	| protected_label 	|
    |:---:	|:------------:	|:-----------------:	|:---------------:	|
    |  0  	|       [0,1,0]      	|         [1,1,0]         	|        0        	|
    |  1  	|       [0,1,1]      	|         [1,0,1]         	|        1        	|
    |  2  	|       [1,1,0]      	|         [1,1,0]         	|        0        	|



#### 3.1 Example: Directory Structure Before Running Calculation:

- `Model_Set`: An example main directory that contains multiple classification model
- `Model_n_n`: An example directory that contains multiple trials of a single model

```bash
metric_calculation/prediction/
    |-- Model_Set_1
    |   |-- Model_1_1
    |   |   |-- try_00.csv
    |   |   `-- try_01.csv
    |   `-- Model_1_2
    |       |-- Model_1_2_1
    |       |   |-- try_00.csv
    |       |   `-- try_01.csv
    |       `-- Model_1_2_2
    |           |-- try_00.csv
    |           `-- try_01.csv
    `-- Model_Set_2
        |-- Model_2_1
        |   |-- try_00.csv
        |   `-- try_01.csv
        `-- Model_2_2
            |-- try_00.csv
            `-- try_01.csv
```



## 4. Calculating confusion matrix
In order to calculate confusion matrix, run the code on `metric_calculation/`:
```bash
python confusion_matrix.py
```

#### 4.1 Example: Directory Structure After Calculating Confusion Matrix

- `cm` directory contains confusion matrix in JSON format.

- `count` directory contains summary of each prediction cases in JSON format.

- Only `cm` directories are used to calculate metrics.

```bash
metric_caluclation/
    |-- Model_Set_1_output
    |   |-- cm
    |   |   |-- Model_1_1
    |   |   |   |-- try_00.json
    |   |   |   `-- try_01.json
    |   |   `-- Model_1_2
    |   |       |-- Model_1_2_1
    |   |       |   |-- try_00.json
    |   |       |   `-- try_01.json
    |   |       `-- Model_1_2_2
    |   |           |-- try_00.json
    |   |           `-- try_01.json
    |   `-- count
    |   |   |-- Model_1_1
    |   |   |   |-- try_00.json
    |   |   |   `-- try_01.json
    |   |   `-- Model_1_2
    |   |       |-- Model_1_2_1
    |   |       |   |-- try_00.json
    |   |       |   `-- try_01.json
    |   |       `-- Model_1_2_2
    |   |           |-- try_00.json
    |   |           `-- try_01.json
    |   |
    `-- Model_Set_2_output
        |-- cm
        |   |-- Model_2_1
        |   |   |-- try_00.json
        |   |   `-- try_01.json
        |   `-- Model_2_2
        |       |-- try_00.json
        |       `-- try_01.json
        `-- count
            |-- Model_2_1
            |   |-- try_00.json
            |   `-- try_01.json
            `-- Model_2_2
                |-- try_00.json
                `-- try_01.json
```

#### 4.2 Example: JSON Confusion Matrix

- 1st level: Protected Group

- 2nd level: Ground Truth (Class)
    
- 3rd level: Confusion Matrix on Classification Result

```json
{
    "0": {
        "0": {"TP": 3, "FP": 2, "TN": 5, "FN": 1},
        "1": {"TP": 5, "FP": 1, "TN": 3, "FN": 2}
    },
    "1": {
        "0": {"TP": 5, "FP": 1, "TN": 2, "FN": 1},
        "1": {"TP": 2, "FP": 1, "TN": 5, "FN": 1}
    }
}
```

Above JSON corresponds to below confusion matrix example:

| Protected group = 0 	| Ground Truth (Class) = 0 	| Ground Truth (Class) = 1 	|
|:-------------------:	|:------------------------:	|:----------------------:	|
|    Prediction = 0   	|             3            	|            2           	|
|    Prediction = 1   	|             1            	|            5           	|


| Protected group = 1 	| Ground Truth (Class) = 0 	| Ground Truth (Class) = 1 	|
|:-------------------:	|:------------------------:	|:----------------------:	|
|    Prediction = 0   	|             5            	|            1           	|
|    Prediction = 1   	|             2            	|            3           	|


## 5. Calculating Metrics

- Confusion matrix must be ready by running `confusion_matrix.py`.

- Run the script at `metric_calculation/` to calculate all metrics:

```bash
./calculate.sh
```

#### 5.1 Example: Directory Structure After Calculating Metrics

- All calculated values are stored in `metric_calculation/Model_Set_output/result/` directory in each model set output directory.

- JSON files: Raw values of each metric calculation

- CSV files: Aggregated values of each metric calculation

```bash
metric_calculation/
    |-- Model_Set_1_output
    |   |-- cm
    |   |-- count
    |   `-- result
    |       |-- bias_amplification.csv
    |       |-- bias_amplification_raw.json
    |       |-- disparate_impact_factor.csv
    |       |-- disparate_impact_factor_raw.json
    |       |-- equality_of_odds_false_positive.csv
    |       |-- equality_of_odds_false_positive_raw.json
    |       |-- equality_of_odds_true_positive.csv
    |       |-- equality_of_odds_true_positive_raw.json
    |       |-- false_positive_subgroup_fairness.csv
    |       |-- false_positive_subgroup_fairness_raw.json
    |       |-- mean_difference_score.csv
    |       |-- mean_difference_score_raw.json
    |       |-- statistical_parity.csv
    |       `-- statistical_parity_raw.json
    `-- Model_Set_2
        |-- cm
        |-- count
        `-- result
            |-- bias_amplification.csv
            |-- bias_amplification_raw.json
            |-- disparate_impact_factor.csv
            |-- disparate_impact_factor_raw.json
            |-- equality_of_odds_false_positive.csv
            |-- equality_of_odds_false_positive_raw.json
            |-- equality_of_odds_true_positive.csv
            |-- equality_of_odds_true_positive_raw.json
            |-- false_positive_subgroup_fairness.csv
            |-- false_positive_subgroup_fairness_raw.json
            |-- mean_difference_score.csv
            |-- mean_difference_score_raw.json
            |-- statistical_parity.csv
            `-- statistical_parity_raw.json
```

#### 5.2 Example: Raw Metric Values in JSON

- Each metric is calculated for individual model sets. 
    - `Model_Set_1` is an example set of models for binary classification task: `class 0` and `class 1`
    - `Model_Set_1` has two models: `Model_1_1` and `Model_1_2`
    - For each model, there are 16 trial results.
    
- 1st level: Model name

- 2nd level: 
    - A list of calculated metric value for each class (`class 0`, `class 1`)
    - Last element: overall value among all classes

```json
{
  "Model_1_1/": [ 
    [0.34759806, 0.47476809, 0.41118307],      % Trial 1
    [0.34759806, 0.47476809, 0.41118307],      % Trial 2
    ...
  ],
  "Model_1_2/": [
    [0.02062884, 0.02176104, 0.02119494],      % Trial 1
    [0.01833622, 0.02196790, 0.02012520],      % Trial 2
    ...
  ]
}
```

#### 5.3 Example: Aggregated Metric Values in CSV

- First column is an aggregated path of a model, which is the path where raw prediction results are stored

- For each classes, 6 values are calculated to aggregate 16 trials for each model.
    - `max_diff`: maximum absolute difference value
    - `max`: maximum value
    - `min`: minimum value
    - `std_dev`: standard deviation value
    - `mean`: mean value
    - `rel_maxdiff(%)`: relative max difference percentage
    - All values are represented in 4 significant digits.
    
    
- `overall` row indicates overall value for all classes

|                     	|          	| max_diff 	| max      	| min       	| std_dev  	| mean     	| rel_maxdiff(%) 	|
|---------------------	|----------	|----------	|----------	|-----------	|----------	|----------	|----------------	|
| ./Model_1/Model_1_1 	| class 0  	| 0.06010  	| 0.1144   	| 0.05432   	| 0.01537  	| 0.06919  	| 86.86          	|
| ./Model_1/Model_1_1 	| class 1  	| 0.03493  	| 0.05119  	| 0.01627   	| 0.01135  	| 0.02943  	| 118.7          	|
| ./Model_1/Model_1_1 	| overall  	| 0.04126  	| 0.07752  	| 0.03626   	| 0.01196  	| 0.04931  	| 83.68          	|
| ./Model_1/Model_1_2 	| class 0  	| 0.1586   	| 0.1636   	| 0.005035  	| 0.04217  	| 0.06067  	| 261.3          	|
| ./Model_1/Model_1_2 	| class 1  	| 0.08165  	| 0.1140   	| 0.03236   	| 0.02139  	| 0.05808  	| 140.6          	|
| ./Model_1/Model_1_2 	| overall  	| 0.1163   	| 0.1388   	| 0.02245   	| 0.03109  	| 0.05938  	| 195.9          	|
