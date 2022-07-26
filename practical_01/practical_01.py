#Design a simple machine learning model to train the training instances and test the same

import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv(r"C:\0_MSc_IT_Notes\Practicals\Machine Learning\practical_01\Cars.csv") #Read data
print(df.head())

plt.scatter(df['Milage'],df['Sell Price']) #Plot to compare Milage(Independant variable) with Sell Price(Dependant variable)
plt.show()
plt.scatter(df['Age'],df['Sell Price'])#Plot to compare Age(Independant variable) with Sell Price(Dependant variable)
plt.show()
X = df[['Milage','Age']] #Determine X
Y = df['Sell Price']    # Determine Y

print("\nindependent varaible x :- \n",X) #Display X
print("\ndependent variable y :- \n",Y) #Display Y

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2)#Create a training set and testing set
print("\nlength of x_train = ",len(X_train))
print("\nlength of X_test =",len(X_test))

from sklearn.linear_model import LinearRegression
clf = LinearRegression()
clf.fit(X_train,Y_train) # Train the model
print("\npredicted X_test :- \n",clf.predict(X_test)) # Use the trained model to predict the testing set

print("\naccuracy = ",clf.score(X_test,Y_test)) #Calculate the accuracy
