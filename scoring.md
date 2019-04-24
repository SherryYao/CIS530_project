## a. formal definition of your metric (2 evaluation metrics are used)
The sentiment labels are:
0 - negative
1 - somewhat negative
2 - neutral
3 - somewhat positive
4 - positive

# a.1.
Performance is evaluated on classification accuracy (the percent of labels that are predicted correctly) for every parsed phrase. 
# a.2. 
Performance is evaluated on the mean squared error.

## b. relevant citations to where it was introduced
(https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews/overview/evaluation)

## c. how to run your evaluation script on the command line (with example arguments, and example output)
python score.py --goldfile 'test_label.csv' --predfile 'lr_test.csv'

Sample output:
"
Namespace(goldfile='test_label.csv', predfile='lr_test.csv')
Accuracy score:  0.5763244224192084
Mean squared error:  0.6545682976198786
"

## d. The scoring.md file should say whether higher scores are better, or lower scores are better
Higher scores are better for both accuracy and mean squared error