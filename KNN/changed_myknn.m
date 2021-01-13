classdef changed_myknn
    methods(Static)
        
        function m = fit(train_examples, train_values, k)
            
			m.mean = mean(train_examples{:,:}); 
			m.std = std(train_examples{:,:}); 
            for i=1:size(train_examples,1)
				train_examples{i,:} = train_examples{i,:} - m.mean; 
                train_examples{i,:} = train_examples{i,:} ./ m.std; 
            end
            
            m.train_examples = train_examples;  
            m.train_values = train_values; 
            m.k = k; 
        
        end

        function predictions = predict(m, test_examples)

            predictions = [];

            for i=1:size(test_examples,1) 
                
                fprintf('classifying example example %i/%i\n', i, size(test_examples,1));
                
                this_test_example = test_examples{i,:}; 
                
                this_test_example = this_test_example - m.mean;
                this_test_example = this_test_example ./ m.std;
                
                this_prediction = changed_myknn.predict_one(m, this_test_example); 
                predictions(end+1) = this_prediction; 
            
            end
        
		end

        function prediction = predict_one(m, this_test_example)
            
            distances = changed_myknn.calculate_distances(m, this_test_example); 
            [neighbour_indices, neighbour_distances] = changed_myknn.find_nn_indices(m, distances); 
            prediction = changed_myknn.make_prediction(m, neighbour_distances, neighbour_indices);
        
        end

        function distances = calculate_distances(m, this_test_example)
            
			distances = [];
            
			for i=1:size(m.train_examples,1) 
                
				this_training_example = m.train_examples{i,:}; 
                this_distance = changed_myknn.calculate_distance(this_training_example, this_test_example); 
                distances(end+1) = this_distance; 
            end
        
		end

        function distance = calculate_distance(p, q) 
            
			differences = q - p; 
            squares = differences .^ 2; 
            total = sum(squares); 
            distance = sqrt(total); 
        
		end

        function [neighbour_indices, neighbour_distances] = find_nn_indices(m, distances) %changed function to return both neighbour_indices and neighbour_distances as both are needed when making a prediction (calculating weighted mean)
            
			[sorted, indices] = sort(distances); 
            neighbour_indices = indices(1:m.k); 
            neighbour_distances = sorted(1:m.k); 
        
		end
        label
        %make a prediction for each test example
        function prediction = make_prediction(m, neighbour_distances, neighbour_indicies)

            %store all the neighbour values using neighbour indices
			neighbour_values = m.train_values(neighbour_indicies); 
            
            %start of weighted mean calculation
            
            %calculate the weights using the distances
            weights = 1./neighbour_distances;
            %calculte some of the weights, used to alter the weights so
            %they add up to one
            weights_sum = sum(1./neighbour_distances);
            %calculate altered weights by diving the weights by there sum
            altered_weights = weights./weights_sum;
         
            %reshape neighbours values to 1*k instead of k*1, this is so
            %they can be multiplied with the altered_weights array
            n_values = reshape(neighbour_values,[1,m.k]);
            
            %multiply re-shaped nieghbour values with the altered weights
            %(these weights are proportional to 1/distance) to get the
            %weighted values then sum the weighted values to get the
            %weighted mean and store in 'prediction' which is returned by
            %the function
            weighted_values = n_values .* altered_weights;
            prediction = sum(weighted_values); 
            
            %end of weighted mean calculation
        
		end

    end
end
