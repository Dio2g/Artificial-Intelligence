classdef myknn
    methods(Static)
        %the 'fit' function performs all the steps involed with training the k-NN classifier 
        %the struct 'm' short for model (which is returned by the 'fit' function) is a data structure created in the 'fit' function which stores all the information about the classifier  
        function m = fit(train_examples, train_labels, k)
            
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
            m.k = k; % stores the k value (parsed into the 'fit' function as a parameter) in the struct 'm'. the k value represents the total amount of nearest neighbours the algorithm will be used during the classification phase
        
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

            for i=1:size(test_examples,1) % loop over number of examples, this is so that a prediction can be made for each test example, the number of examples is equal to the height of the examples table found using the size function
                
                fprintf('classifying example example %i/%i\n', i, size(test_examples,1)); % print current example being classified so the user can see the progress of the algorithm
                
                this_test_example = test_examples{i,:}; % set 'this_test_example' to the current test example in the loop, this is so that the example can be standardized and a prediction can be made
                
                % start of standardisation process, re-scales the values of
                % each feature in the training examples to make sure they lie
                % within similar ranges, this is crucial as if features have a
                % large range of values they may dominate the distance
                % calculations, so re-scaling them will prevent this
                
                % apply z-score stardarisation to just this test example
                this_test_example = this_test_example - m.mean;
                this_test_example = this_test_example ./ m.std;
                
                % end of standardisation process
                
                this_prediction = myknn.predict_one(m, this_test_example); % call the predict_one function giving the struct 'm' and current test example in the loop as its parameters
                predictions(end+1) = this_prediction; % append the result of the predict_one function to the end of predictions to build the list of predictions for the test examples
            
            end
        
		end

        % the 'predict_one' function makes one prediction for one test
        % example, it does this by calling the 'calculate_distances'
        % function on the test example which returns all the distances
        % between the test example and all the training examples, then it
        % calls the 'find_nn_indices' function which returns all the
        % indices corresponsing with the first k distances, after that the
        % 'make_prediction' is called which find the labels of the training
        % data corresponsing with those indices and returns a prediction (the mode of the labels)
        function prediction = predict_one(m, this_test_example)
            
            distances = myknn.calculate_distances(m, this_test_example); % call calculate distaces giving the struct 'm' and the test examples as parameters, this is so we can calculate the distance between the current test example and all the training examples
            neighbour_indices = myknn.find_nn_indices(m, distances); % call 'find_nn_indices' function giving 'm' and 'distances' (calculated in the line above) as parameters. this is so we can find all the nieghbour indices corresponsing with the first k distances, this is used later to find the labels for each example
            prediction = myknn.make_prediction(m, neighbour_indices); %call the 'make_prediction' function giving 'm' and the previously found 'neighbour_indices' as paramets, this function makes a prediction for the current test example, by taking the mode of its first k neighbours labels
        
        end

        %The 'calculate_distances' function calculates the Euclidean straight-line
        %distances between the current test example and all the training
        %examples and then returns these distances in the form of an array,
        %it goes this by calling the 'calculate_distance' function on each
        %training example
        function distances = calculate_distances(m, this_test_example)
            
            %define distances as an empty array
			distances = [];
            
			for i=1:size(m.train_examples,1) %loop through each training example so that the distance can be calcuated between each example and the current test example
                
				this_training_example = m.train_examples{i,:}; % set 'this_training_example' to the current training example in the loop
                this_distance = myknn.calculate_distance(this_training_example, this_test_example); %call 'calculate_distance' function giving current training example and current test exmaple as parameters and store in this_distance, this will give the distance between the current test example and the current training example
                distances(end+1) = this_distance; %append this distance to end of array of distances, when loop is complete this array will have each distance between the current test example and all of the training examples stored within it, the number of examples is equal to the height of the examples table found using the size function
            end
        
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

        %the 'find_nn_indices' function returns k indices of the closest k
        %distances, the indices of the distances also correspond to the
        %indices of the training examples
        function neighbour_indices = find_nn_indices(m, distances)
            
			[sorted, indices] = sort(distances); %sort distances so that the smallest distance is in the first position 
            neighbour_indices = indices(1:m.k); %store all the indices of the sorted distances from index 1 to k (k is stored in the struct 'm')
        
		end
        
        %makes a prediction for a single test example, it does this by
        %taking all the labels of its k nearest neighbours and then return
        %the mode (most common) label
        function prediction = make_prediction(m, neighbour_indices)

			neighbour_labels = m.train_labels(neighbour_indices); %store labels of all the neigbour indices
            prediction = mode(neighbour_labels); %set prediction as the most common label stored
        
		end

    end
end
