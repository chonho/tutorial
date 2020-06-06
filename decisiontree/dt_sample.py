from sklearn.datasets import load_iris
from sklearn import tree
import pydotplus 
 
if __name__ == "__main__":
     
    #loading iris dataset
    iris = load_iris()
     
    clf = tree.DecisionTreeClassifier()
    clf.fit(iris.data, iris.target)
     
    dot_data = tree.export_graphviz(clf, out_file=None,  
                         feature_names=iris.feature_names,  
                         class_names=iris.target_names,  
                         filled=True, rounded=True,  
                         special_characters=True) 
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_png("iris.png")
