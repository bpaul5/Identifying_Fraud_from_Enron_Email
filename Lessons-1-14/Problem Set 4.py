### Lesson 4 Choose Your Own Algorithm 

#k nearest neighbors - uses a distance function to classify new cases, k should be odd to avoid ties, 
#if then value can be chosen at random, larger value of k usually leads to a more precise value. 

from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(features_train, labels_train)
pred = neigh.predict(features_test)
accuracy_score(pred, labels_test)


adaboost

random forest 

