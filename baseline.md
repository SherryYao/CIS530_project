## how to run the code
Python script to run the program
python simple-baseline.py --trainfile 'train_data.csv' --testfile 'test_data.csv' --magnitudeFile 'glove.6B.100d.magnitude' --outputDirectory 'lr_test.csv'
(The file 'glove.6B.100d.magnitude' is used in Homework4. It is too large to upload to the repo, so we didn't upload it.)

Linear classifier used: 1. Logistic Regression; 2. Random Forest; 3. GBDT; 4. NN

## 1. Logistic Regression
## parameter tuning using validation set:
C = [0.001, 0.005, 0.01, 0.05, 0.1, 1, 1.5]
Acc = 
[0.5514911622193408,
 0.5527195796082713,
 #0.5532655428922405, (max)
 0.5531972974817444,
 0.5529925612502559,
 0.5528560704292637,
 0.5528560704292637]
## Acc score for test set (with C = 0.01):
0.5483

## 2. Random Forest
## parameter tuning using validation set:
# a. n_estimator: number of decision tree
n_estimator = [20, 40, 60, 80, 100, 120, 140, 180, ...]
Acc = 
[
    0.5278782501876749,
    0.5437111854227803,
    0.5465092472531222,
    0.5476694192315567,
    0.5491025728519757,
    0.5524465979662868,
    0.5548351873336518,
    0.5588616665529244,
    ...
]
The larger the n_estimator, the higher the accuracy, but we couldn't increase the n_estimator all the time because of huge computational complexity.

# b. criterion: The function to measure the quality of a split.
criterion = [Gini (default), Information Gain]
Acc = [0.5588616665529244, 0.5514229168088446]

## Acc score for test set (with n_estimator = 180 & criterion = Gini)
0.547567529838766

## 3. GBDT
still working on it

## 4. NN
(# hidden layers, # neurons in each layer)
(3, 1), (5, 1), (10, 1),(3, 2), (10, 2), (3, 3)

Accuracy score:
[
    0.551627653040333,
    0.5462362656111377,
    0.5529925612502559, (max)
    0.5383880434040811,
    0.555858868491094,
    0.5521736163243022
] 
## Acc score for test set (with 10 hidden layers and 1 neurons per layer)
0.5523836113631605
