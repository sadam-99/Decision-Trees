from DecisionTree import *
import pandas as pd
from sklearn import model_selection
leaf_depth= []
inner_prune_ids=[]

pruned_id = []
# header = ['SepalL', 'SepalW', 'PetalL', 'PetalW', 'Class']
# header = ['Age of patient at time of operation', 'Patients year of operation','Number of positive axillary nodes detected' , 'Survival status']
header = ['Sample code number','Clump Thickness,Uniformity of Cell Size','Uniformity of Cell Shape,Marginal Adhesion','Single Epithelial Cell Size','Bare Nuclei','Bland Chromatin','Normal Nucleoli','Mitoses','Class']
# df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/haberman/haberman.data', header=None, names= ['Age of patient at time of operation', 'Patients year of operation','Number of positive axillary nodes detected' , 'Survival status'])
# df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None, names=['SepalL','SepalW','PetalL','PetalW','Class'])
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data', header=None, names=['Sample code number','Clump Thickness,Uniformity of Cell Size','Uniformity of Cell Shape,Marginal Adhesion','Single Epithelial Cell Size','Bare Nuclei','Bland Chromatin','Normal Nucleoli','Mitoses','Class'])
lst = df.values.tolist()
t = build_tree(lst, header)
print_tree(t)


print("********** Leaf nodes ****************")
leaves = getLeafNodes(t)
for leaf in leaves:
    # dict_leaf[leaf.depth]=leaf.id
    leaf_depth.append(leaf.depth)
    print("id = " + str(leaf.id) + " depth =" + str(leaf.depth))
 
largest_depth= max(leaf_depth)
print("********** Non-leaf nodes ****************")
innerNodes = getInnerNodes(t)
for inner in innerNodes:
    print("id = " + str(inner.id) + " depth =" + str(inner.depth))
    if inner.depth >= largest_depth-2:
        inner_prune_ids.append(inner.id)
# if largest_depth-2 == inner.depth:
# pruned_id.append(inner.id)
        
# Splitting the training and testing data
trainDF, testDF = model_selection.train_test_split(df, test_size=0.2)
train = trainDF.values.tolist()
test = testDF.values.tolist()

t = build_tree(train, header)
print("*************Tree before pruning*******")
print_tree(t)
acc_bef_prun = computeAccuracy(test, t)
print("Accuracy on test before pruning = " + str(acc_bef_prun))

## TODO: You have to decide on a pruning strategy

# pruned_id = get_pruned_id(t)
# t_pruned = prune_tree(t, [26, 11, 5])
for i in range(4):
    pruned_id = ext_random_pruned_ids(inner_prune_ids, 0.60)
    print(inner_prune_ids)
    print(pruned_id)
	## TODO: You have to decide on a pruning strategy

t_pruned = prune_tree(t, pruned_id)

print("*************Tree after pruning*******")
print_tree(t_pruned)
acc_aft_prun = computeAccuracy(test, t_pruned)
print("Accuracy on test after pruning = " + str(acc_aft_prun))

# Checking if the accuracy is improved or not
if acc_aft_prun>acc_bef_prun:
    print("Accuracy is improved by %d ", (acc_aft_prun-acc_bef_prun))
else:
    print("Accuracy is not improved")

    
