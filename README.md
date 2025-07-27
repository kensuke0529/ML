# Credit Card Fraud Detection

### Overview
This project applies several machine learning models to detect credit card fraud dataset. 

### Dataset
Credit Card Fraud Detectio: 
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

### Summery
 XGBoost showed the top performer with a precision of 93% and a recall of 85%. In the context of fraud detection, reducing false negatives is critical, as missing a fraudulent transaction has a high cost. The model achieves a strong balance, demonstrating both accuracy and robustness.

###  Summary Table

| Model                  | Precision | Recall | F1-Score |
|------------------------|-----------|--------|----------|
| Logistic Regression  | 0.69      | 0.84   | 0.76     |
| Random Forest          | 0.82      | 0.80   | 0.81     |
| XGBoost                | 0.98      | 0.82   | 0.89     |
| Isolation Forest       | 0.30      | 0.43   | 0.36     |
| PyTorch                | 0.80      | 0.81   | 0.80     |

### 1. LogisticRegression.ipynb
- Implemented as a baseline model using `class_weight='balanced'` to address severe class imbalance.  
- Threshold tuning was performed to optimize the trade-off between detecting frauds (Recall) and minimizing false alarms (Precision)

- Based model:
    ```
    [[55519  1345]   # TN, FP
     [    8    90]]  # FN, TP

                  precision    recall  f1-score   support
    Class 0 (Legit)     1.00      0.98      0.99     56864
    Class 1 (Fraud)     0.06      0.92      0.12        98
    ```
   While recall for the fraud was 0.92, which is very high, the number of false positives (FP) was also very high (1345). This means the model was detecting most fraud but also misclassifying a large number of normal transactions as fraud.


- Tuned model
    ```
    [[56828    36]   
    [   16    82]]  

                  precision    recall  f1-score   support
    Class 0 (Legit)     1.00      1.00      1.00     56864
    Class 1 (Fraud)     0.69      0.84      0.76        98
    ```
    After tuning the thresholds, false positives were reduced from 1345 → 36, while still detecting most of the actual fraud cases (90 → 82).

### 2. RandomForest.ipynb
* **Initial Model (Base Random Forest):**
    * Utilized `class_weight='balanced'` to handle the highly imbalanced dataset. The base model showed a  strong base performance:
        ```
        [[56863     1]
         [   24    74]]
                      precision    recall  f1-score   support

        Class 0 (Legit)     1.00      1.00      1.00     56864
        Class 1 (Fraud)     0.99      0.76      0.86        98
        ```
    * This initial model achieved an excellent **F1-score of 0.86** for the fraud class, with high Precision (0.99) and decent Recall (0.76). Critically, it produced only **1 False Positive**. However, it **missed 24 actual frauds (False Negatives)**, which could be a concern if avoiding missed fraud is the primary business objective.

* **Optuna-Tuned & Threshold-Optimized Model:**
    * To further enhance performance and optimize for specific business requirements, hyperparameters were tuned using **Optuna**, followed by an adjustment of the prediction threshold based on the Precision-Recall curve.
        ```
        [[56847    17]
         [   20    78]]
                      precision    recall  f1-score   support

        Class 0 (Legit)     1.00      1.00      1.00     56864
        Class 1 (Fraud)     0.82      0.80      0.81        98
        ```
    * While it resulted in slightly more **False Positives (17 vs. 1)** compared to the base model, it **caught more actual frauds (78 vs. 74 True Positives)**. 
    
    * This is trade-off of accepting a small increase in false alarms to significantly improve the detection rate of actual fraud cases **aligns better with typical business requirements** where minimizing financial loss from undetected fraud is paramount.

**3. XGBoost Performance:**

* **Base XGBoost:**
    * trained using `scale_pos_weight`. Its initial performance on the test set was pretty strong:

        ```
        [[56863     1]
         [   19    79]]
                      precision    recall  f1-score   support

        Class 0 (Legit)     1.00      1.00      1.00     56864
        Class 1 (Fraud)     0.99      0.81      0.89        98
        ```
    * This base model achieved an outstanding **F1-score of 0.89** for the fraud class, with exceptionally high **Precision (0.99)** and robust **Recall (0.81)**. It detected **79 True Positives** while producing only **1 False Positive**.

* **Tuned Model:**


        [[56862     2]
         [   18    80]]

                precision    recall  f1-score   support

            0       1.00      1.00      1.00     56864
            1       0.98      0.82      0.89        98



    * This fine-tuning maintained the model's high performance, demonstrating XGBoost's inherent strength and its ability to achieve top-tier results in fraud detection scenarios.
    * That model is overall best model in terms of f1-score which means balance detecting frauds and avoiding the false alarms. 


**4. Isolation Forest Performance:**

* **Base Isolation Forest:**
    * The Isolation Forest model, configured with `contamination` rate derived from the training data, was used for anomaly detection. 

        ```
        Confusion Matrix:
        [[56798    66]
         [   66    32]]

        Classification Report:
                      precision    recall  f1-score   support

        Class 0 (Legit)     1.00      1.00      1.00     56864
        Class 1 (Fraud)     0.33      0.33      0.33        98
        ```
    * The base Isolation Forest model yielded an **F1-score of 0.33** for fraud detection, with 66 FP and 66 FN.

* **Optuna-Tuned Model:**
   

        Confusion Matrix:
        [[56768    96]
         [   56    42]]

        Classification Report:
                      precision    recall  f1-score   support

        Class 0 (Legit)     1.00      1.00      1.00     56864
        Class 1 (Fraud)     0.30      0.43      0.36        98

    * While tuning slightly improved the F1-score and recall, the Isolation Forest still demonstrated limitations in this specific fraud classification task compared to the supervised learning approaches, particularly due to a higher number of False Positives.
 

**5. Pytorch:**
* The Isolation Forest model, configured with `contamination` rate derived from the training data, was used for anomaly detection. 


        Confusion Matrix:
        [[56844    20]
        [   19    79]]
        
        Classification Report:
            precision    recall  f1-score   support

        0       1.00      1.00      1.00     56864
        1       0.80      0.81      0.80        98

    Performed similarly to Random Forest; less optimal than XGBoost 




