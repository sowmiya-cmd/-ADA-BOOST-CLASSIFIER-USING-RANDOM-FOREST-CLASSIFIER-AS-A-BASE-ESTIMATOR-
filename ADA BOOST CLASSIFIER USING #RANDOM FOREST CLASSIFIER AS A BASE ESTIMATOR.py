#ADA BOOST CLASSIFIER USING #RANDOM FOREST CLASSIFIER AS A BASE ESTIMATOR 

from sklearn.ensemble import AdaBoostClassifier

base_cls = RandomForestClassifier() 
from sklearn import metrics
# Create adaboost classifer object

abc = AdaBoostClassifier(n_estimators=50,base_estimator = base_cls,
                         learning_rate=1)

# Train Adaboost Classifer
model = abc.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = model.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
