# python3
from sklearn import tree

feature = [[178,1], [155,0], [177,0], [165,0], [169,1], [160,0]]
label = ['male','female','male','female','male','female']

clf = tree.DecisionTreeClassifier()

clf.fit(feature, label)

clf.predict([[168,0]])


print(clf.predict([[168,0]]))