# Making k-NN Model and Decision Tree Classifier using the MNIST dataset <br>
Link to Jupyter notebook (on google colab): https://colab.research.google.com/drive/14u99p-dFeXcXiXCuOewLhYfx3vMsEdLP?usp=sharing
<br> <br>
Loading MNIST, data splitting and class distributions
Firstly, the MNIST dataset was imported and loaded using the TensorFlow Keras API. Then, the data was flattened to make them 1-dimensional, so that they can be used to make the k-NN model and the decision tree classifier. 80% of the data was used to train the models while 20% was used in the validation set. The test set was taken as is, but it was flattened so it can be used to see how well the models perform later on. The class distribution was computed and plotted in subplots for each of the datasets, as shown in the figures below:
<br><br>
![](https://github.com/zeniamazhar/CS412-assignment/blob/main/ValidationClassDistribution.png)

Figure 1: Bar chart showing the number of the samples against the digits that were handwritten in the samples in the validation set. The class distribution for each digit was as follows: {0: 1175, 1: 1322, 2: 1174, 3: 1219, 4: 1176, 5: 1104, 6: 1177, 7: 1299, 8: 1160, 9: 1194}
<br>
![](https://github.com/zeniamazhar/CS412-assignment/blob/main/TestClassDistribution.png)

Figure 2: Bar chart showing the number of the samples against the digits that were handwritten in the samples in the test set. The class distribution for each digit was as follows: {0: 980, 1: 1135, 2: 1032, 3: 1010, 4: 982, 5: 892, 6: 958, 7: 1028, 8: 974, 9: 1009}
<br>
![](https://github.com/zeniamazhar/CS412-assignment/blob/main/TrainingClassDistribution.png)

Figure 3: Bar chart showing the number of the samples against the digits that were handwritten in the samples in the training set. The class distribution for each digit was as follows: {0: 4748, 1: 5420, 2: 4784, 3: 4912, 4: 4666, 5: 4317, 6: 4741, 7: 4966, 8: 4691, 9: 4755}

<br><br>

It can be seen that the class distributions are fairly uniform for each of the digits in each of the datasets. Hence, it is safe to continue with these splits and there isn’t any bias that would be caused by uneven distribution of the numbers of samples for particular digits in any of the datasets. 

Basic statistics and Data Preprocessing

Some basic statistics were computed for each of the three datasets, including the mean values (of the x values) and the standard deviations. As can be inferred from the subplots of the class distributions shown earlier, the mean values and standard deviations were similar for each of the three datasets, which provided further confirmation that there is little to no bias when it comes to the class distributions of the numbers of samples. Table 1 below summarizes the values found for each of these, and it can be seen that the standard deviations were around 78, while the mean values were around 33 for each of the datasets.




Table 1: A table summarizing the basic statistical values computed for all three datasets, including the mean and the standard deviation.


|                    | Training set | Validation set | Test set |
|--------------------|-------------|---------------|---------|
| **Mean**          | 33.34       | 33.23         | 33.79   |
| **Standard deviation** | 78.59       | 78.46         | 79.17   |

Subsequently, sample images were shown for each digit by making subplots (Figure 4). The first occurrence of each digit in the training set was used for this purpose, while showing the digits in 2 rows, with 5 digits in each row (See Figure 4).

<br>

![](https://github.com/zeniamazhar/CS412-assignment/blob/main/Samples.png)

Figure 4: Subplots showing sample images of each digit (1st occurrence of the digit in the training set). 

<br>


This was followed by preprocessing the data by first finding the minimum and maximum pixel values of the training data, which were found to be 0 and 255, respectively. All the x-values in the three datasets were then normalized by being divided by 255 to normalize the data points so they can be in the 0-1 range.

Hyperparameter tuning of k 
The validation set was used to find the best value of k out of these options: 1, 3, 5, 7, and 9. This was done by first making an array of these values, and then iterating through each of them one by one and initializing the number of neighbors of a k-NN classifier with the current value of k, followed by training the model on the training dataset. The accuracy was then calculated for the predictions made by the k-NN model for the validation set, and this value was then appended to a list. The number of neighbors which gave the highest accuracy (97.41%) was selected as being 1 (See Table 2). 

This was an unexpected outcome, given that the k-NN model is expected to memorize the training data when the number of neighbors is limited to being 1, so the model was not expected to perform the best when predicting the numbers in the validation set due to not being able to generalize to new data points. However, it is possible that the data sets are very similar, as evident by the mean values of the pixel intensities (Table 1). This may make it so there are misclassifications that are caused by taking into account the labels of multiple data points, and the model isn’t able to predict as accurately when it comes to the validation set. This may also mean that the boundaries between the different number labels are very clear, and that more neighbors aren’t required to make accurate predictions. 

| Number of Neighbors (k) | Accuracy (%) |
|-------------------------|-------------|
| 1                       | 97.41       |
| 3                       | 97.27       |
| 5                       | 97.15       |
| 7                       | 96.96       |
| 9                       | 96.73       |


Table 2: A table showing the number of neighbors used to initialize the k-NN model and the accuracy computed for its predictions of the labels on the validation set.

Moreover, a plot was made to show the trend in the validation accuracy against the number of neighbors used to initialize the k-NN model (See Figure 5). It can be seen that as the number of neighbors increases, the validation accuracy goes down. This isn’t seen commonly, but this may be due to the same reason the best performance was seen when the k-NN model had k=1: The training set and the validation set may have been very similar, and the boundaries between the different classes may be very clear. 

<br>

![](https://github.com/zeniamazhar/CS412-assignment/blob/main/ValidationAccuracykNN.png)

Figure 5: Line graph showing a visual representation of the decrease in the accuracy of the predictions of the k-NN models against the number of neighbors (k) used to initialize the models.

The final k-NN model with k=1
The final k-NN classifier was made with k=1 as this was the best value found for k as seen in the hyperparameter tuning step. Predictions were made for the test set by using this classifier, after which the accuracy score was computed, which was found to be 96.73%, which is slightly lower than the accuracy found for the validation set predictions (97.41%). However, this was expected, since the test set is completely separate from the training set, unlike the validation set. In addition to this, the precision, recall, and the F1-score values were also computed, which were found to be 0.97. 

A confusion matrix was generated for the final k-NN model (Figure 6). The most frequently misclassified numbers seem to be the following (in order of highest to lower number of misclassifications): (1) 4 misclassified as 9 (25 occurrences), (2) 3 misclassified as 5 (21 occurrences), (3) 8 misclassified as 5 (18 occurrences). Intuitively, these misclassifications make sense - 4 can indeed resemble 9 if written in a certain manner, as they are only different by a few pixels at the top. Similarly, 5 can be written in a way where it resembles a 3 but without the pixels that make up the top line of the number 3. 5 and 8 may be confused with each other, given that sometimes 8 is handwritten in a slanted manner where the top half of the digit looks like a continuous line rather than a circle, making it look like the number 5. Moreover, the fact that only one nearest neighbor is being used to predict the labels, the digits that are written in a more “unique” manner would likely be misclassified. Additionally, the model’s predictions are very sensitive to noise as well (due to k=1, and the classifier being a k-NN model), making it so even the digits that are “clearly” written would be misclassified.

<br>

![](https://github.com/zeniamazhar/CS412-assignment/blob/main/ConfusionMatrixkNN.png)

Figure 5: Confusion matrix generated for the k-NN model’s (with k=1) predictions on the test set. 

<br>

5 random samples of misclassified handwritten digits were generated, and are shown in figure 6 below. It can be seen that 2 of these predictions were 9 whereas the true label was 4 - which was the case for most of the misclassified cases. For the example with the true label being 5 and the prediction being 6, and the example with the true label being 8 and the prediction being 7, shows that the model has a difficult time when the difference between the digits is only a few pixels - just a few pixels being filled at the bottom left would’ve turned the 5 into a 6, and the erasure of a few pixels from the top left of the 8 would make it look like a 7.

Hyperparameter tuning for Decision Tree Classifier 
Next, the decision tree classifier was built by first tuning the minimum samples split (min_samples_split) and the maximum depth (max_depth). The possible values for the max_depth were 2, 5, and 10. The possible values for the min_samples_split were 2 and 5. All possible combinations of each of these hyperparameters were checked (by initializing decision trees to the hyperparameters and training them on the training set) to see which combination yields the highest accuracy when making predictions on the validation set. The min_samples_split of 5 and max_depth of 10 gave the highest accuracy (85.79%), and this was selected for the final decision tree classifier. 

Final Decision Tree Classifier 
The final decision tree classifier was initialized with max_depth = 10 and min_samples_split=5, and trained using the training set. The predictions of this tree for the test set was found to have an accuracy of 86.43%, which is higher than the accuracy obtained for the validation set. The precision, recall, and F1-score were all 0.86. Already, it can be seen that this model performed worse than the k-NN model - and this may be due to the nature of the data. The data consists of handwritten digits, and it makes intuitive sense to use the nearest neighbors to predict the label of the digits. However, it is much more difficult to ask questions regarding an attribute of the image and split the data according to the answer, especially since the maximum depth has been set to a specific value. The model may have performed better if the maximum depth was set to a higher value, but this would also bring the risk of overfitting/memorization of the training set. If the data was, instead, more numerical/hierarchical, the decision tree classifier might’ve performed better than the k-NN model, as the attributes would split the data according to different ranges a specific attribute (or a set of attributes) falls into. 

Following this, a confusion matrix was generated for the final decision tree classifier (Figure 6). The misclassifications with the highest frequencies were the following: (1) 4 misclassified as 9 (94 occurrences), (2) 5 misclassified as 3 (59 occurrences), (3) 8 misclassified as 9 (49 occurrences). Interestingly, this classifier had the same misclassification as the most frequent one, as we saw with the k-NN model - 4 being misclassified as 9. This is quite intuitive, as the two numbers look alike, and have the difference of only a few pixels at the top, which none of the attributes that were used to make the classifications might have been able to filter through. The second highest misclassification is the reverse of the second highest misclassification we saw for the k-NN model, and the resemblance between these digits may be in the same way, where the top line written for 3 may be the only difference between a 5 and a 3, causing the model to frequently misclassify 5 as 3. The fact that 8 is misclassified as 9 very frequently also makes intuitive sense, due to the difference only being of a few pixels between 8 and 9 in the bottom left corner of the digits. This may mean that the decision tree is filtering the digits by taking one “line” of digits as the attribute, which would explain the similar pattern seen in the most frequent misclassifications.

Finally, a ROC curve was generated for each digit in the test set in the decision tree classifier, and the AUC score was calculated for each digit (Figure 7). The AUC is the area under the curve, and it represents how well the model can differentiate between the different classes, and a number closer to 1 means the decision tree classifier is able to differentiate between the digit from the other digits well. It can be seen in figure 7 that the classifier performs well on predicting the true labels of the test data, as the AUC scores are above 0.90 for all of the digits. It has the most difficult time differentiating 8 from the other digits (AUC = 0.91), which is also evident from the fact that it misclassified 8 as 9 very frequently (as mentioned earlier). On the other hand, this classifier is able to tell the difference between 1 and other digits the best (with AUC = 0.99). 

![](https://github.com/zeniamazhar/CS412-assignment/blob/main/ROCDC.png)

Figure 7: ROC curve for each digit in the test set for the predictions made by the decision tree classifier with min_samples_split = 5 and max_depth = 10.
