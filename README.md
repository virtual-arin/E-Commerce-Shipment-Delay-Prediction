# E-Commerce Shipment Delay Prediction 🚚📦

## 😓 The Problem & Business Impact
* **The Problem:** A large e-commerce warehouse struggles with delayed deliveries and cannot accurately predict which specific orders will arrive late. 
* **Business Impact:** By proactively identifying high-risk orders, the business can alert customers early, prioritize specific shipments, optimize warehouse workflows, and ultimately improve customer satisfaction.

## 📌 Project Overview
Built a machine learning project to predict whether an e-commerce order will be delivered on time or delayed. By predicting delays before they happen, logistics teams can take proactive action to keep customers happy.

## 🎯 Objective
Build a **Supervised Binary Classification** model to classify the delivery status:
* **0** = On-time delivery
* **1** = Delayed delivery

## 📈 Dataset 
* **Source:** [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce?resource)
* **Important Note:** The `olist_order_reviews_dataset.csv` was explicitly excluded from this project to prevent **data leakage** (since reviews happen *after* delivery).

## 🛠️ Methodology
1. **Data Preparation & EDA:** Merged multiple datasets (customers, orders, products, sellers, etc.) to consolidate information. Initial exploratory analysis revealed a highly imbalanced dataset where only ~7.9% of shipments were actually delayed.
2. **Data Preprocessing:** Handled missing values by imputing with robust metrics (median for numerical, mode for categorical) and leveraged estimated delivery dates where actual dates were missing. Removed extreme numerical outliers using the Interquartile Range (IQR) method.
3. **Feature Engineering:** Extracted meaningful patterns, such as calculating geographical distances (`lat_diff`, `lng_diff`), deriving time components (`processing_time`, `purchase_hour`), and defining `weight_category` variables.
4. **Encoding & Scaling:** Applied Ordinal Encoding for weight classes and One-Hot Encoding for customer and seller states. Scaled all numerical features using `StandardScaler` to ensure optimal algorithm performance.
5. **Model Training & Hyperparameter Optimization:** Evaluated a diverse set of baseline models including Logistic Regression, K-Nearest Neighbors (KNN), Decision Tree, and AdaBoost. To improve predictive performance, an extensive hyperparameter optimization process was executed on a LinearSVC using `GridSearchCV`. 

## 🏆 Final Results
`Naive Bayes` achieved the highest recall `99.04%`, but at the cost of extremely low precision (8.07%) and low accuracy (11.17%), resulting in too many false alarms. Therefore, Support Vector Machine was selected as the final model because it provided a better balance between identifying delayed orders and minimizing unnecessary interventions, achieving a recall of 68% and the highest F1-score of 0.30.

## 💻 Technologies Used
* **Language:** Python
* **Data Manipulation & Visualization:** Pandas, NumPy, Matplotlib, Seaborn
* **Machine Learning:** Scikit-Learn (Classification Models, `StandardScaler`, `RandomizedSearchCV`)
* **Environment:** Jupyter Notebook

## 🚀 Future Improvements
* **Integrate External Data:** Add real-time weather or traffic conditions to improve prediction accuracy.
* **Try Advanced Algorithms:** Experiment with gradient boosting models like XGBoost or LightGBM.
* **Deployment:** Wrap the final model in a simple web API (using Flask or FastAPI) so the software can be used in real-time.