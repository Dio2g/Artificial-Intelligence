classdef mykdtree
    methods(Static)
        %the 'fit' function performs all the steps involed with training the kd-tree classifier 
        %the struct 'm' short for model (which is returned by the 'fit' function) is a data structure created in the 'fit' function which stores all the information about the classifier 
        function m = fit(train_examples, train_labels)
            
            % start of standardisation process, re-scales the values of
            % each feature in the training examples to make sure they lie
            % within similar ranges, this is crucial as if features have a
            % large range of values they may dominate the distance
            % calculations, so re-scaling them will prevent this
			m.mean = mean(train_examples{:,:}); % calulate the mean of the training examples and store in the struct 'm', this is used in the z-score standardisation calculation
			m.std = std(train_examples{:,:}); % calculate standard deviation of training examples, this is used in the z-score standardisation calculation
            for i=1:size(train_examples,1) % loop over each value in train examples so that the z-score standardisation can be done for each one
				% apply z-score stardarisation to each example
                train_examples{i,:} = train_examples{i,:} - m.mean; % subtract the mean from the example
                train_examples{i,:} = train_examples{i,:} ./ m.std; % divide by the stadard deviation
            end
            % end of standardisation process

            m.train_examples = train_examples; % takes a copy of all training examples (parsed into the 'fit' function as a parameter) and stores them in the struct 'm'
            m.train_labels = train_labels; % takes a copy of all training labels (parsed into the 'fit' function as a parameter) and stores them in the struct 'm'
            %m.k = k;

            %create empty node structure
            emptyNode.axis = [];
            emptyNode.location = [];
            emptyNode.left_child = [];
            emptyNode.right_child = [];
            
            m.emptyNode = emptyNode; % takes a copy of emptyNode and stores in the struct 'm'
            
            %convert examples to cell array
            m.train_examples = table2cell(m.train_examples);

            %for testing
            %point_list = [{7, 2}; {5, 4}; {9, 6}; {4, 7}; {8, 1}; {2, 3}];
            
            %generate kd-tree by calling kdtree() function giving the
            %training examples and a depth of 0 to start
            tree = mykdtree.kdtree(m.train_examples, 0);
            m.tree = tree;
            
            %display tree to check everything is working properly
            %mykdtree.displayTree(tree);
            
            %for testing I used data and output from the wiki
            %data: 'point_list = [{7, 2}; {5, 4}; {9, 6}; {4, 7}; {8, 1};
            %{2, 3}];'
            %correct output: '((7, 2), ((5, 4), ((2, 3), None, None), ((4, 7), None, None)), ((9, 6), ((8, 1), None, None), None))'
            %upon testing this worked correctly

        end
        
        %function to generate a kd tree given examples and a default depth
        %of 0
        function node = kdtree(examples, depth)
            %if examples are empty then return an empty node
            if isempty(examples)
                node = [];
                return;
            end
            
            k = length(examples(1,:)); %assumes all points have the same dimensions
            axis = mod(depth, k); %choose axis depending on depth
            node.axis = axis+1; %set node axis to axis + 1 as matlab arrays start at 1 and not 0
            examples = sortrows(examples, axis+1); %sort examples by axis (+ 1 as matlab arrays start at 1 and not 0)
            median = floor(height(examples)/2)+1; %calculate median from sorted examples (used as pivot point)

            %create node and construct subtrees
            node.location = examples(median,:); %set location at median
            %node.label = labels(median);
    
            node.left_child = mykdtree.kdtree(examples(1:median-1,:), depth+1); %construct left child

            node.right_child = mykdtree.kdtree(examples(median+1:end,:), depth+1); %construct right child
        end
        
        %nearest neighbours search function for generated
        %kd-tree from the kdtree function
        function nearestNeighbours(node, point, closest, min_distance)
            global closeArr; %array of closest nodes
            if(~isempty(node)) %make sure node is not empty
                if (isempty(node.right_child)&&isempty(node.left_child)) %checks to see if node is a leaf node
                   distance = mykdtree.calculate_distance(cell2mat(point), cell2mat(node.location)); %calculate Euclidean straight-line distance between given point and current node
                   if (distance < min_distance) %checks if calculated distance is less than store minimum distance
                       min_distance = distance; %update min distance
                       closest = node; %update closest node                     
                       closeArr(end+1) = closest; %append closest node to closest node global array
                   end
                else %if not a leaf node
                    %descend tree recursively
                   if point{node.axis} < node.location{node.axis} %check if point is less than current node in the split dimension
                       %search left first
                       mykdtree.nearestNeighbours(node.left_child, point, closest, min_distance)
                       if (point{node.axis} + min_distance) >= node.location{node.axis} %check if point+mininmum distance is greater than or equal to urrent node in the split dimension
                           %search right
                           mykdtree.nearestNeighbours(node.right_child, point, closest, min_distance);
                       end
                   else %if point is greater than current node in the split dimension
                       %search right first
                       mykdtree.nearestNeighbours(node.right_child, point, closest, min_distance)
                       if (point{node.axis} - min_distance) <= node.location{node.axis} %check if point-mininmum distance is less than or equal to urrent node in the split dimension
                           %search left
                           mykdtree.nearestNeighbours(node.left_child, point, closest, min_distance);
                       end
                   end
                end
            end
        end

        %function to display generate kd-tree from kdtree function, used
        %for testing
        function displayTree(tree)
            if isempty(tree)
            else
                tree
                mykdtree.displayTree(tree.left_child);
                mykdtree.displayTree(tree.right_child);        
            end
        end

        
        %the 'predict' function performs all the steps required in
        %predicting each label for the test examples, it takes the model
        %'m' create earlier in the 'fit' function and also the
        %'test_examples as its parameters, this function basically loops
        %through each example in 'test_examples' and calls the predict_one
        %function on each one, then it appends the returned predictions to
        %a 'predictions' array
        function predictions = predict(m, test_examples)

            predictions = categorical; % defines predictions as a categorical array as the predictions array will store the predicted labels for each test example, this is non-numerical data

            for i=1:size(test_examples,1)  % loop over number of examples, this is so that a prediction can be made for each test example, the number of examples is equal to the height of the examples table found using the size function
                
                fprintf('classifying example example %i/%i\n', i, size(test_examples,1)); % print current example being classified so the user can see the progress of the algorithm
                
                this_test_example = test_examples{i,:}; % set 'this_test_example' to the current test example in the loop, this is so that the example can be standardized and a prediction can be made
                
                % start of standardisation process, re-scales the values of
                % each feature in the training examples to make sure they lie
                % within similar ranges, this is crucial as if features have a
                % large range of values they may dominate the distance
                % calculations, so re-scaling them will prevent this
                
                % start of standardisation process
                this_test_example = this_test_example - m.mean;
                this_test_example = this_test_example ./ m.std;
                % end of standardisation process
                
                this_prediction = mykdtree.predict_one(m, this_test_example); % call the predict_one function giving the struct 'm' and current test example in the loop as its parameters
                predictions(end+1) = this_prediction; % append the result of the predict_one function to the end of predictions to build the list of predictions for the test examples
            
            end
        
		end

        %this function generates a prediction for a single test example
        function prediction = predict_one(m , this_test_example)
            
            %create global struct array, used to store closest nodes
            global closeArr;
            closeArr = struct('axis', {}, 'location', {}, 'left_child', {}, 'right_child', {});
            
            %convert current test examples to cell array
            this_test_example = num2cell(this_test_example);
            
            %call nearest neighbour function to find nearest neighbour in
            %the kdtree to the
            %current test example 
            mykdtree.nearestNeighbours(m.tree,this_test_example,m.emptyNode,intmax('uint64'));
            
            %for testing nearest neighbours I used this test point:
            %'test_point = {-1.2535, -0.1623, -1.3150, -1.1637}'
            %it seemed to work correctly
            %I have not implemented k nearest neighbours yet, only nearest
            %neighbours search
            
            %nearest neighbour is located at position 1 of the closest node
            %array
            loc = cell2mat(closeArr(1).location);
            index = [];
            %find index of label for the found nearest neighbour by looping
            %over training examples and finding index of equal training
            %example to nearest neighbour
            for i=1:size(m.train_examples,1)
                currentRow = m.train_examples(i,:);
                temp = cell2mat(currentRow);
                if temp == loc
                    index = i; %set index to index of current row
                end
            end
           
            prediction = m.train_labels(index); %set prediction for thius current test example usinjg found index of label
        end

        

        %The 'calculate_distance' function calculates the Euclidean straight-line
        %distance between any 2 examples given as parameters using Pythagoras' Theorem
        function distance = calculate_distance(p, q) 
            
            %start of distance calculation
			differences = q - p; %calculate differences between each of the points 
            squares = differences .^ 2; %square the differences
            total = sum(squares); %sum all the squares
            distance = sqrt(total); %squareroot the sum of the square to give the distance
            %end of distance calculation
        
        end

    end
end


