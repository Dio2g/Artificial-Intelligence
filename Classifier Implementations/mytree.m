classdef mytree
    methods(Static)
        %the 'fit' function performs all the steps involed with training the DT classifier
        function m = fit(train_examples, train_labels)
            
            %defines a re-useable empty node structure which can be copied
            %to represent individual nodes in the tree
            
            %the (unique) number of the nodes within the overall tree structure
			emptyNode.number = [];
            
            %any training examples the node holds
            emptyNode.examples = [];
            %any associated labels the node holds
            emptyNode.labels = [];
            
            %a prediction based on any class labels the node holds
            emptyNode.prediction = [];
            
            %a numeric measure of the impurity of any class labels held by a node (used in deciding whether to split it)
            emptyNode.impurityMeasure = [];
            
            %if the decision is taken to split a node then it will store two child nodes and divide its training data between them
            emptyNode.children = {};
            
            
            %the particular feature which define the split
            
            %number of its column
            emptyNode.splitFeature = [];
            %name
            emptyNode.splitFeatureName = [];
            %value
            emptyNode.splitValue = [];
            

            m.emptyNode = emptyNode;
            
            %copies empty node structure to create the root node of the
            %tree
            r = emptyNode;
            %set root node number to 1 as it is first node in the tree
            r.number = 1;
            %copy ALL training labels
            r.labels = train_labels;
            %copy ALL training examples
            r.examples = train_examples;
            
            %generate a single class label prediction for the node, used to
            %classify new data if it isn't possible to split the root node
            %any further
            r.prediction = mode(r.labels);
            

            %set up othe model parameters 
            
            %minimum number of examples the nodes must contain before
            %considering to split them 
            m.min_parent_size = 10;
            %list of all unique class labels
            m.unique_classes = unique(r.labels);
            %names of all the individual features within each example
            m.feature_names = train_examples.Properties.VariableNames;
            %current number of nodes in the tree
			m.nodes = 1;
            %total number of training examples used to train the model
            m.N = size(train_examples,1);
            
            %Generate the tree. Parses root node to the 'trySplit'
            %function, this function test to see whether a node can be
            %split into 2 child nodes with reduced overall impurity wihthin
            %their respective class labels. This function is also recursive
            %meaning it will call itself again giving the new child nodes
            %as parameters, it will only return once its not possible to
            %split any more nodes
            m.tree = mytree.trySplit(m, r);

        end
        
        function node = trySplit(m, node)

            %check to make sure the node is large enough to be split and
            %become a 'parent', the minimum requirement for this is set in
            %the 'fit()' function
            if size(node.examples, 1) < m.min_parent_size
                %if the number of examples in the node is not large enough
                %the function returns imediately 
				return
			end

            %calcualte measure of current impurity within the nodes class
            %labels by calling the 'weightedImpurity()' function
            node.impurityMeasure = mytree.weightedImpurity(m, node.labels);

            %looks at possible ways of splitting the training data in the
            %current node
            
            %loops over every feature
            for i=1:size(node.examples,2)

				fprintf('evaluating possible splits on feature %d/%d\n', i, size(node.examples,2));
                
                % reorder the examples, and the labels, based on this feature:
				[ps,n] = sortrows(node.examples,i); %sort current feature values store in 'ps' and index in 'n', index 'n' is used later to sort labels
                ls = node.labels(n); %sort labels to match sorted feature values using previous sorted index, store in 'ls'
                
                %initialize biggest reduction variables, used later to keep
                %track of reduction in impurity and compare the largest impurity reductions
                biggest_reduction(i) = -Inf;
                biggest_reduction_index(i) = -1;
                biggest_reduction_value(i) = NaN;
                
                %loops over unique values of each feature
                for j=1:(size(ps,1)-1) %loops from 1 up to the height of ps minus 1, this is ebcause the number of possible splits is always one less than the number of examples
                    if ps{j,i} == ps{j+1,i} %checks to see if the next value is the same as the current one and skips to the next iteration if its the same, this prevents splitting on the same value more than once
                        continue;
                    end
                    
                    this_reduction = node.impurityMeasure - (mytree.weightedImpurity(m, ls(1:j)) + mytree.weightedImpurity(m, ls((j+1):end))); %calculates the GDI for the two collections of class labels created by a the split, they are then added together and then subtracted them from the GDI of the original table
                    %if the result of this calculation is positive this
                    %means the split is good as it has produced a reduction
                    %in impurity
                    
                    %keeps track of the largest reduction in impurity for
                    %each individual feature, this is because there are
                    %many different splits that may produce reductions in
                    %impurity but we need to keep track of the greatest
                    %reduction
                    if this_reduction > biggest_reduction(i)
                        biggest_reduction(i) = this_reduction; %stores the reduction
                        biggest_reduction_index(i) = j; %stores the index
                    end
                end
				
            end

            %compares the largest impurity reductions achieved across each
            %of the different features and finds the largest one by using
            %the 'max()' function
            [winning_reduction,winning_feature] = max(biggest_reduction);
            winning_index = biggest_reduction_index(winning_feature); %finds the index of the largest reduction 

            if winning_reduction <= 0 %checks the winning reduction to see if its greater than 0, this is so we can work out whether to split the node or not
                return %if is less or equal to 0 then no split happens and the node is returned 
            else%if winning reduction is greater than 0 then split node

                %make two new child nodes, descending directly from the original node, with one containing the first portion of training data, and the other containing the second portion
                
                %sort examples based on winning feature, need to sort again as it is still sorted based on the values in the final feature column
                [ps,n] = sortrows(node.examples,winning_feature);
                ls = node.labels(n);

                
                %set up previously empty structure fields
                
                %the index of the table column containing the feature we have decided to split the training data on
                node.splitFeature = winning_feature;
                
                %the name of the table column containing the feature we have split the training data on, used to print out in text-based visualisation of the tree
                node.splitFeatureName = m.feature_names{winning_feature};
                
                %the value of the feature that should be used for splitting
                %future testing data, choose a value half way between the
                %unqiue training value we split on and the next highest value of that feature in the training data
                node.splitValue = (ps{winning_index,winning_feature} + ps{winning_index+1,winning_feature}) / 2;

                
                node.examples = []; %delete the training examples from the current node as they will be contained in the node's children
                node.labels = []; %delete the training labels from the current node as they will be contained in in the node's children
                node.prediction = []; %delete the prediction associated with this node as it will not be used to generate a prediction later during the prediction phase
                %this is all done to remove duplicated data and save space

                %create a new empty node for the first child
                node.children{1} = m.emptyNode;
                m.nodes = m.nodes + 1; %increment number of nodes in the tree as we have just created a new one
                node.children{1}.number = m.nodes; %unique number for the node within the overall tree
                node.children{1}.examples = ps(1:winning_index,:); %all the rows from the parent's table of training data which have a value in the splitFeature column less than or equal to the splitValue
                node.children{1}.labels = ls(1:winning_index); %the class labels associated with the examples in each child node
                node.children{1}.prediction = mode(node.children{1}.labels); %most common class label in the labels, used as a prediction
                
                %create a new empty node for the second child
                node.children{2} = m.emptyNode;
                m.nodes = m.nodes + 1; %increment number of nodes in the tree as we have just created a new one
                node.children{2}.number = m.nodes; %unique number for the node within the overall tree
                node.children{2}.examples = ps((winning_index+1):end,:); %all the rows from the parent's table of training data which have a value in the splitFeature column greater than the splitValue
                node.children{2}.labels = ls((winning_index+1):end); %the class labels associated with the examples in each child node
                node.children{2}.prediction = mode(node.children{2}.labels); %most common class label in the labels, used as a prediction
                
                %recursive call to 'trySplit()' function on the 2 created
                %child nodes to see if they can be split to further reduce the
                %overall impurity
                node.children{1} = mytree.trySplit(m, node.children{1});
                node.children{2} = mytree.trySplit(m, node.children{2});
            end

        end
        
        %calculate GDI -> g(L) = 1 - SIGMA Pc^2
        function e = weightedImpurity(m, labels)

            weight = length(labels) / m.N; %calculate weight, this models the probability of arriving at a particular node when decending the tree from the root node, needed to make fair comnparisons between impurity of single parent nodes VS 2 potential child nodes

            summ = 0; %(SIGMA)/SUM
            obsInThisNode = length(labels);
            for i=1:length(m.unique_classes) %loop over every unqiue class label (SIGMA)
                
				pc = length(labels(labels==m.unique_classes(i))) / obsInThisNode; %calculate fraction of class labels (Pc) in given labels (L) which belong to current class being iterated over 
                summ = summ + (pc*pc); %square pc and add to running total  (Pc^2)
            
			end
            g = 1 - summ; %(1 - SIGMA Pc^2)
            
            e = weight * g; %factor in the weigh calculated earlier 

        end

        %the 'predict' function performs all the steps required in
        %predicting each label for the test examples
        function predictions = predict(m, test_examples)

            predictions = categorical; %defines predictions as a categorical array as the predictions array will store the predicted labels for each test example, this is non-numerical data
            
            for i=1:size(test_examples,1) %loop over number of examples, this is so that a prediction can be made for each test example, the number of examples is equal to the height of the examples table found using the size function
                
				fprintf('classifying example %i/%i\n', i, size(test_examples,1)); %print current example being classified so the user can see the progress of the algorithm
                this_test_example = test_examples{i,:}; %set 'this_test_example' to the current test example in the loop, this is so that a prediction can be made for the current example
                this_prediction = mytree.predict_one(m, this_test_example); %call the predict_one function giving the struct 'm' and current test example in the loop as its parameters
                predictions(end+1) = this_prediction; %append the result of the predict_one function to the end of predictions to build the list of predictions for the test examples
            
			end
        end

        %the 'predict_one' function 
        function prediction = predict_one(m, this_test_example)
            
			node = mytree.descend_tree(m.tree, this_test_example); %return winning leaf node by calling 'decend_tree()' on current example
            prediction = node.prediction; %return prediction found in previously returned winning node
        
		end
        
        %the 'descend_tree' function applies the split rules defined in
        %each node by looking at its splitFeature and splitValue fields and
        %comparing against the value of the corresponding feature in the
        %current test example, until it reaches a leaf node, then it
        %returns that node
        function node = descend_tree(node, this_test_example)
            
			if isempty(node.children) %if a leaf node is reached return the node
                return;
            else
                if this_test_example(node.splitFeature) < node.splitValue %if the value of the example with the same feature as the nodes split feature is less than the nodes split value 
                    node = mytree.descend_tree(node.children{1}, this_test_example); %call decend_tree() the first child node
                else %if the value of the example with the same feature as the nodes split feature is greater than or equal too the nodes split value 
                    node = mytree.descend_tree(node.children{2}, this_test_example);%call decend_tree() the second child node
                end
            end
        
		end
        
        % describe a tree:
        function describeNode(node)
            
			if isempty(node.children)
                fprintf('Node %d; %s\n', node.number, node.prediction);
            else
                fprintf('Node %d; if %s <= %f then node %d else node %d\n', node.number, node.splitFeatureName, node.splitValue, node.children{1}.number, node.children{2}.number);
                mytree.describeNode(node.children{1});
                mytree.describeNode(node.children{2});        
            end
        
		end
		
    end
end