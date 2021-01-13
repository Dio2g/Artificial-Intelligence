classdef mynb
    methods(Static)
        %the 'fit' function performs all the steps involed with training the NB classifier
        %the 'fit' function does 2 main things, it estimates the
        %probability density for the distribution of each feature within
        %each class (by calculating the means and stds for each one of these features) and it estimates a prior probability for each class based on how many times the class label occurs in the training data
        %the struct 'm' short for model (which is returned by the 'fit'
        %function) is a data structure created in the 'fit' function which
        %stores all the unique classes and also the means and standard
        %deviations for each feature of each unique class
        function m = fit(train_examples, train_labels)

            m.unique_classes = unique(train_labels); %find all different possible class labels by calling the unique() function on the labels from the training data, unique classes are needed here so they can be iterated over later 
            m.n_classes = length(m.unique_classes); %find out how many different class there are by calling the length() function on the unique classes, length is needed so a loop that iterates over each unqiue class can be built

            m.means = {}; %initialize 'means' variable as a cell array
            m.stds = {}; %initialize 'stds' variable as a cell array
            
            for i = 1:m.n_classes %loops over each unique class
            
                % set up the this_class variable as an example, this_class
                % stores the current label being iterated over from all the
                % unique training labels
				this_class = m.unique_classes(i);
                
                % find all the rows in 'train_examples'
                % where the corresponding element of train_labels was 
                % equal to the value stored inside the this_class variable
                % using logical indexing (finds all examples with same
                % label), these examples are needed so that we can
                % calculate the mean and std for each feature wihtin the
                % example, the means and stds are then used to estimate the
                % probability densitys
                examples_from_this_class = train_examples{train_labels==this_class,:};
                
                %calculates the mean for each feature (used to calculate the normal distribution) by calling the mean()
                %function on the 2D array 'examples_from_this_class' which
                %contains all the values of all the features corresponding
                %label being iterated over in the loop, this array of means
                %is then appended onto the cell array 'means', a cell array
                %is used so that the means of all the features
                %corresponding which each different label can be stored
                m.means{end+1} = mean(examples_from_this_class);
                %calculates the standard deviation for each feature (used to calculate the normal distribution) by calling the std()
                %function on the 2D array 'examples_from_this_class' which
                %contains all the values of all the features corresponding
                %label being iterated over in the loop, this array of
                %standard deviations
                %is then appended onto the cell array 'stds', a cell array
                %is used so that the standard deviations of all the features
                %corresponding with each different label can be stored
                m.stds{end+1} = std(examples_from_this_class);
                
                %both the means and stds are needed here in order to estimate a probability density function for the distribution of each feature within each class
            
			end
            
            m.priors = []; %intialise 'priors' variable as array
            
            for i = 1:m.n_classes %loops over all the values in the 'unique_classes' for the second time
                
                % set up the this_class variable as an example, this_class
                % stores the current label being iterated over from all the
                % unique training labels
				this_class = m.unique_classes(i);
                % find all the rows in 'train_examples'
                % where the corresponding element of train_labels was 
                % equal to the value stored inside the this_class variable
                % using logical indexing (finds all examnples with same label)
                examples_from_this_class = train_examples{train_labels==this_class,:};
                
                %calculate prior current label by dividing the amount of
                %examples from current class by the amount of total
                %examples, append this to the 'priors' array, an array is
                %used here so we can store the prior for each unqiue label,
                %a prior is an estimate of how likely each class label is to occur, based on how many times it is present in the training data
                m.priors(end+1) = size(examples_from_this_class,1) / size(train_labels,1);
            
			end

        end

        %the 'predict' function performs all the steps required in
        %predicting each label for the test examples
        function predictions = predict(m, test_examples)

            predictions = categorical; % defines predictions as a categorical array as the predictions array will store the predicted labels for each test example, this is non-numerical data

            for i=1:size(test_examples,1) % loop over number of examples, this is so that a prediction can be made for each test example, the number of examples is equal to the height of the examples table found using the size function

				fprintf('classifying example %i/%i\n', i, size(test_examples,1)); % print current example being classified so the user can see the progress of the algorithm
                this_test_example = test_examples{i,:}; % set 'this_test_example' to the current test example in the loop, this is so that a prediction can be made for the current example
                this_prediction = mynb.predict_one(m, this_test_example); % call the predict_one function giving the struct 'm' and current test example in the loop as its parameters
                predictions(end+1) = this_prediction; % append the result of the predict_one function to the end of predictions to build the list of predictions for the test examples
            
			end
        end

        %the 'predict_one' function makes a single prediction given one
        %test example, it does this by looping over all the possible class labels and, for each one, calculating a likelihood for the current test example given the class
        function prediction = predict_one(m, this_test_example)

            for i=1:m.n_classes %loops for each unique class label, this is so a likelihood of the current test example (to be within a given class) can be calculated for each unqiue class

				this_likelihood = mynb.calculate_likelihood(m, this_test_example, i); %calls 'calculate_likelihood' function on current test example with current unqiue class label, also passing the model 'm' as a parameter, the returned calculated likelihood for the current test example is then stored in 'this_likelihood'
                this_prior = mynb.get_prior(m, i); %get prior of current class from model 'm'
                posterior_(i) = this_likelihood * this_prior; %calculate posterior for current example in current class by multiplying the examples likelihood by the classes prior, the calculated psoterior values are guranteed to be proportinal to the true posterior probabilities
            
			end

            [winning_value_, winning_index] = max(posterior_); %get index of the winning value by calling max() on the calculated posterior, the calculated psoterior values are guranteed to be proportinal to the true posterior probabilities, this means by calling the max() function on the array we can reliably select the most likely class label
            prediction = m.unique_classes(winning_index); %store prediction by indexing the unique classes with the winning idex (of max posterior)

        end
        
        %the 'calculate_likelihood' function calculates the likelihood of
        %the given test example to be within the given class
        function likelihood = calculate_likelihood(m, this_test_example, class)
            
			likelihood = 1; %set to 1 so that when multiplication (to calc liklihood) happens a value is still given instead of null or 0.
            
			for i=1:length(this_test_example) %loop over every feature in the curent test example
                likelihood = likelihood * mynb.calculate_pd(this_test_example(i), m.means{class}(i), m.stds{class}(i)); %calculate likelihood by calculating the probability density of every feature in the current test example and then multiplying these all together to get single estimate of the likelihood of the example given the class, it is possible to calculate the likelihood this way due to the class-conditional independence assumption
            end
        end
        
        %get prior given model 'm' and class index
        function prior = get_prior(m, class)
            
			prior = m.priors(class);
        
		end
        
        function pd = calculate_pd(x, mu, sigma) %calculates probability density 
        
            %normal distribution formula
			first_bit = 1 / sqrt(2*pi*sigma^2);
            second_bit = - ( ((x-mu)^2) / (2*sigma^2) );
            pd = first_bit * exp(second_bit);
        
		end
            
    end
end