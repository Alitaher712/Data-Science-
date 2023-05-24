import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
data = pd.read_csv(r'Churn_Modelling.csv')
#########Data Wranglanging
#checking for missing values
print(data.isna().any())
print(data.nunique())
#check if there a duplicated customer id
print(data.duplicated('CustomerId').sum())
print("--------------------------- Describe Atributes")
print(data.describe())
print(data.info())
print("------------------------------------------------")
#drop nin necassricy colmns
dataset = data.drop(["RowNumber", "CustomerId", "Surname"], axis = 1)
dataset.head()
df1=data.copy()
print(df1.head())
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df1['Gender_encoded'] = le.fit_transform(df1['Gender'])
df1.drop('Gender',axis=1, inplace=True)
df1.head()
#visualization
#data vislization
#1)basic visualization to understand how the data is distributed(histogram,piechart)
count = dataset["Exited"].value_counts()

plt.figure(figsize=(15, 6))

#Count plot
plt.subplot(1, 2, 1)
sns.countplot(x='Exited', data=dataset)
plt.title("Count Plot of Exited")
plt.xlabel("Exited")
plt.ylabel("Count")
plt.xticks([0, 1], ['Retained', 'Exited'])

# Pie chart
plt.subplot(1, 2, 2)
labels = 'Exited', 'Retained'
sizes = [count[1], count[0]]
explode = (0, 0.1)

plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
shadow=True, startangle=90)
plt.axis('equal')
plt.title("Proportion of customer Exited and retained", size=14)

plt.tight_layout()
plt.show()

# From the visualization above, the number of customers that exited the bank is lower compared to the number of
# customers that didn’t leave the bank.









#2)visualize the relationship between the target variable (exited) and gender attribute
plt.figure(figsize=(10, 6))

plt.subplot(1, 2, 1)
sns.countplot(x='Gender', data=dataset, hue='Exited', palette='Set2')
plt.title('Gender')

plt.subplot(1, 2, 2)
sns.countplot(x='Geography', data=dataset, hue='Exited', palette='Set2')
plt.title('Geography')

plt.tight_layout()
plt.show()

# Gender->From the visualization above, Female customers left the bank more often compared to the Male customers
# Geography->From the visualization above, the average loss of customers is highest in Germany followed by France and
# the least in Spain


#Multiple box plots
fig, axarr = plt.subplots(3, 2, figsize=(14, 6))
sns.boxplot(y='CreditScore',x = 'Exited', hue = 'Exited',data = dataset, ax=axarr[0][0])
sns.boxplot(y='Age',x = 'Exited', hue = 'Exited',data = dataset , ax=axarr[0][1])
sns.boxplot(y='Tenure',x = 'Exited', hue = 'Exited',data = dataset, ax=axarr[1][0])
sns.boxplot(y='Balance',x = 'Exited', hue = 'Exited',data = dataset, ax=axarr[1][1])
sns.boxplot(y='NumOfProducts',x = 'Exited', hue = 'Exited',data = dataset, ax=axarr[2][0])
sns.boxplot(y='EstimatedSalary',x = 'Exited', hue = 'Exited',data = dataset, ax=axarr[2][1])
plt.show()

#Tenure=>Number of years for which the customer has been with the bank
sns.displot(data = data, x ='Tenure', kind = 'hist', color = 'g')
plt.show()
#we see that customers has been with the bank about 1 to 8 on averge





#3)The relationship between ‘Age’ and the target variable (Exited).
plt.figure(figsize = (8,6))
sns.lineplot(x = "Age", y = "Exited", data = dataset)
plt.show()
# From the visualization above, exited customers are older, on average, than those still active. This kind of makes sense, as
# clients who have left must have been with the bank some time. The young ones have not really had the reason or the opportunity
# to yet leave(Most of the customers who leave the bank are between 40 and 60 years)

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import SGDRegressor

x = df1.drop(["RowNumber", "CustomerId", "Surname", "Exited", "Geography"], axis=1)
y = df1["Exited"]

# normalize X
scaler = StandardScaler()
x = scaler.fit_transform(x)
#Split the data

x, X_test, y, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
x.shape, X_test.shape, y.shape, y_test.shape

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error, accuracy_score,make_scorer

print("Linear Regression:")
model1 = SGDRegressor(loss='squared_error', max_iter=100)

# Train the model
model1.fit(x, y)

# Make predictions on the training and testing sets
y_train_pred = model1.predict(x)
y_test_pred = model1.predict(X_test)

#Note that we added .round() to the predicted values to convert them into binary predictions before calculating the accuracy score.
print("Training MAE:", mean_absolute_error(y, y_train_pred))
print("Testing MAE:", mean_absolute_error(y_test, y_test_pred))
print("Training MSE:", mean_squared_error(y, y_train_pred))
print("Testing MSE:", mean_squared_error(y_test, y_test_pred))
print("Training accuracy:", accuracy_score(y, y_train_pred.round())*100,"%")
print("Testing accuracy:", accuracy_score(y_test, y_test_pred.round())*100,"%","\n")





# Define the hyperparameters to tune
model = SGDRegressor()
# Define the hyperparameters to tune and their possible values
param_grid = {
'loss': ['squared_error', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
'alpha': [0.0001, 0.001, 0.01],
'learning_rate': ['constant', 'invscaling', 'adaptive'],
'max_iter': [500, 1000 , 2000]
}

#cv=5 In 5-fold cross-validation, the dataset is divided into 5 equal-sized folds. The model is trained and evaluated 5
# times, each time using a different fold as the validation set and the remaining folds as the training set.

#n_jobs: This parameter specifies the number of jobs or CPU cores to use for parallel computation.
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2',n_jobs=-1)
grid_search.fit(x, y)

print("Best parameters: ", grid_search.best_params_)
from sklearn.metrics import mean_absolute_error, accuracy_score,make_scorer
#apply best parameter to model
best_model = grid_search.best_estimator_
y_train_pred = best_model.predict(x)
y_test_pred = best_model.predict(X_test)
y_pred_rounded = y_test_pred.round()
accuracy = accuracy_score(y_test, y_pred_rounded)
print("testing Accuracy score: ", accuracy*100,"%")
print("Training accuracy score:", accuracy_score(y, y_train_pred.round())*100,"%")
# bonus
# Define hyperparameters
def fit_model(x, y, epochs, alpha):
    alpha = alpha
    num_iters = epochs

    # Initialize the coefficients
    theta = np.zeros(x.shape[1])
    theta0 = 0

    # Define the cost function Mean Squared Error (MSE)
    def cost_function(x, y, theta, theta0):
        m = len(y)
        h = x.dot(theta) + theta0
        J = (1 / (2 * m)) * np.sum(np.square(h - y))
        return J

    # Define the gradient descent function
    def gradient_descent(x, y, theta, theta0, alpha, num_iters):
        m = len(y)
        J_history = []
        for i in range(num_iters):
            h = x.dot(theta) + theta0
            theta = theta - (alpha / m) * np.sum(x.T.dot(h - y))
            theta0 = theta0 - (alpha / m) * np.sum((h - y))
            J_history.append(cost_function(x, y, theta,theta0))
            print(cost_function(x, y, theta,theta0))
        return theta, theta0, J_history

    # Apply Gradient Descent
    theta, theta0, J_history = gradient_descent(x, y, theta, theta0, alpha, num_iters)

    # Predict values
    y_pred = x.dot(theta) + theta0

    # Calculate the mean squared error
    mse = mean_squared_error(y, y_pred)
    print("BONUS..........")
    print("Mean squared error:", mse)
    # Calculate the accuracy score
    accuracy = (1 - mse) * 100
    print("Accuracy score:", accuracy, "%")


# Training data
fit_model(x, y, 1000, 0.01)
# testin data
fit_model(X_test, y_test, 1000, 0.01)