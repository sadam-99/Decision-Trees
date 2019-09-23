# driver.py-
        ## Variables.
                - 'header' : Attributes Names
                - df : dataframe for loading data
                - t: Tree
                - largest_depth: maximum depth of the leaf.
                - inner_prune_ids:  the IDs of the Inner Nodes
                - acc_bef_prun= Accuracy before Pruning
                - t_pruned: Pruned tree
                - acc_aft_prun=  Accuracy After Pruning
        ## Functions:
                - build_tree: Builds the tree   
                - getLeafNodes: Returns List of leaf nodes(IDs)
                - getInnerNodes: Returns List of Internal nodes(IDs)
                - computeAccuracy: Computed the accuracy of the model
                - ext_random_pruned_ids: extract the random samples of inner nodes(IDs) using some sampling value
                - prune_tree- Pruns the tree

# Datasets Used for testing-
        ## Iris.data, haberman.data, breast-cancer-wisconsin.data, hepatitis.data
       

# How to run the code:
        ## Run command python driver.py on anaconda prompt.
        


