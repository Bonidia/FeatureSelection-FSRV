# A Novel Decomposing Model with Evolutionary Algorithms for Feature Selection in Long Non-Coding RNAs


## Authors

* Robson P. Bonidia, Jaqueline Sayuri Machida, Tatianne C. Negri, Wonder A. L. Alves, André Y. Kashiwabara, Douglas S. Domingues, André C.P.L.F. de Carvalho, Alexandre R. Paschoal, Danilo S. Sanches

* **Correspondence:** rpbonidia@gmail.com or bonidia@usp.br or danilosanches@utfpr.edu.br


## Publication

If you use this code in a scientific publication, we would appreciate citations to the following paper:

Submitted


## List of files

 - **Datasets:** Datasets;

 - **GA-CFS-ACC** Decomposing Model with Genetic Algorithm (Fitness = CFS and ACC) - Python;
 
 - **GA-CFS** Decomposing Model with Genetic Algorithm (Fitness = CFS (Filter Approach - Main)) - Python;
 
 - **GA-wrapper** Decomposing Model with Genetic Algorithm (Wrapper approach) - Python;
 
 - **PSO-wrapper** Decomposing Model with Particle Swarm Optimization (Wrapper approach) - Python;

 - **README:** Documentation;

 - **Requirements:** List of items to be installed using pip install.
 
 - **split_train_test** Split dataset into training and testing - Python;


## Dependencies

- Python (>=3.7.4)
- NumPy 
- Pandas
- Scikit-learn
- Skfeature-chappers


## Installing our tool

```sh
$ git clone https://github.com/Bonidia/FeatureSelection-FSRV.git FeatureSelection-FSRV

$ cd FeatureSelection-FSRV

$ pip3 install -r requirements.txt
```

## Usange and Examples

## Split dataset into training and testing

Firstly, it is necessary to separate the dataset in training and testing. We will only use the training set for feature selection. The test set will be used to generate a final report with the efficiency of the best feature subset.

```sh
Access folder: $ cd FeatureSelection-FSRV
 
To run (Example): $ python3.7 split_train_test.py -i input -r test_rate

Where:

-i - input - csv format file, e.g., dataset.csv

-r - TEST_RATE - e.g., 0.2, 0.3
```


This example will generate a training and test file.

**Note:** Input samples for feature selection must be in csv format.

**Dataset:** It is important that the csv file contains the following format: feat1, feat2, ..., featk, label - **The label/class must be the last column.**


**Running**

```sh
python3.7 split_train_test.py -i lncRNA.csv -r 0.2
```


## **GA-CFS** Decomposing Model with Genetic Algorithm (Fitness = CFS (Filter Approach - Main))

```sh
Access folder: $ cd FeatureSelection-FSRV
 
To run (Example): $ python3.7 GA-CFS.py -train training.csv -test testing.csv -classifier classifier

Where:

-train - csv format file (training set), e.g., train.csv

-test - csv format file (testing set), e.g., test.csv

-classifier - e.g., 0 = RandomForestClassifier, 1 = DecisionTreeClassifier, 2 = SVM, 3 = KNN, 
                    4 = GaussianNB, 5 = GradientBoosting, 6 = Bagging, 7 = AdaBoost, 8 = MLP
```

This example will generate a csv file with the selected features.

**Note 1:** Input samples for feature selection must be in csv format.

**Note 2:** In this algorithm, the classifier will be used to generate the final report.

**Note 3:** We will only use the training set for feature selection. 

**Note 4:** The test set will be used to generate a final report with the efficiency of the best feature subset.


**Running**

```sh
python3.7 GA-CFS.py -train training.csv -test testing.csv -classifier 2
```

## **GA-CFS-ACC:** Decomposing Model with Genetic Algorithm (Fitness = CFS and ACC - Hybrid)

```sh
Access folder: $ cd FeatureSelection-FSRV
 
To run (Example): $ python3.7 GA-CFS-ACC.py -train training.csv -test testing.csv -classifier classifier

Where:

-train - csv format file (training set), e.g., train.csv

-test - csv format file (testing set), e.g., test.csv

-classifier - e.g., 0 = RandomForestClassifier, 1 = DecisionTreeClassifier, 2 = SVM, 3 = KNN, 
                    4 = GaussianNB, 5 = GradientBoosting, 6 = Bagging, 7 = AdaBoost, 8 = MLP
```


This example will generate a csv file with the selected features.

**Note 1:** Input samples for feature selection must be in csv format.

**Note 2:** We will only use the training set for feature selection. 

**Note 3:** The test set will be used to generate a final report with the efficiency of the best feature subset.


**Running**

```sh
python3.7 GA-CFS-ACC.py -train training.csv -test testing.csv -classifier 2
```

## **GA-wrapper** Decomposing Model with Genetic Algorithm (Wrapper approach)

```sh
Access folder: $ cd FeatureSelection-FSRV
 
To run (Example): $ python3.7 GA-wrapper.py -train training.csv -test testing.csv -classifier classifier

Where:

-train - csv format file (training set), e.g., train.csv

-test - csv format file (testing set), e.g., test.csv

-classifier - e.g., 0 = RandomForestClassifier, 1 = DecisionTreeClassifier, 2 = SVM, 3 = KNN, 
                    4 = GaussianNB, 5 = GradientBoosting, 6 = Bagging, 7 = AdaBoost, 8 = MLP
```

This example will generate a csv file with the selected features.

**Note 1:** Input samples for feature selection must be in csv format.

**Note 2:** We will only use the training set for feature selection. 

**Note 3:** The test set will be used to generate a final report with the efficiency of the best feature subset.


**Running**

```sh
python3.7 GA-wrapper.py -train training.csv -test testing.csv -classifier 2
```


## **PSO-wrapper** Decomposing Model with Particle Swarm Optimization (Wrapper approach)

```sh
Access folder: $ cd FeatureSelection-FSRV
 
To run (Example): $ python3.7 PSO-wrapper.py -train training.csv -test testing.csv -classifier classifier

Where:

-train - csv format file (training set), e.g., train.csv

-test - csv format file (testing set), e.g., test.csv

-classifier - e.g., 0 = RandomForestClassifier, 1 = DecisionTreeClassifier, 2 = SVM, 3 = KNN, 
                    4 = GaussianNB, 5 = GradientBoosting, 6 = Bagging, 7 = AdaBoost, 8 = MLP
```

This example will generate a csv file with the selected features.

**Note 1:** Input samples for feature selection must be in csv format.

**Note 2:** We will only use the training set for feature selection. 

**Note 3:** The test set will be used to generate a final report with the efficiency of the best feature subset.


**Running**

```sh
python3.7 PSO-wrapper.py -train training.csv -test testing.csv -classifier 2
```


## About

If you use this code in a scientific publication, we would appreciate citations to the following paper:

Submitted
