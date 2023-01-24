## Credit card fraud prediction

- The objective of this project is to develop a model that detects whether a credit card transaction is `Fraud` or `Not Fraud`
- This is a binary classification problem. The class is highly imbalanced: 99% of the transactions are `Not Fraud` and only 1% is `Fraud`.
- The challenge of working with imbalanced datasets is that most machine learning techniques will ignore, which in turn leads to poor performance on, the minority class, although typically it is performance on the minority class that is most important (in this case, detecting fraud transactions)
- We will apply resampling methods (e.g., SMOTE, Undersampling the majority class) to the train dataset, employ Cost-sensitive learning, and select the appropiate evaluation metrics for the model.  
