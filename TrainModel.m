% The code we used to apply PCA analysis, and train the SVM, DT and KNN models and also evaluated
% their performance using leave-one-subject-out cross validation.
all_data = cell(1, 10);
for patient = 1:10
    results_filename = sprintf('Patient_%02d_FeatureMatrix.mat', patient);
    load(results_filename, 'feature_matrix');
    all_data{patient} = feature_matrix;
end

% Combine all features for PCA computation
combined_features = [];
for patient = 1:10
    combined_features = [combined_features; all_data{patient}(:, 1:34)];
end

standardized_features = combined_features - mean(combined_features);
standardized_features = standardized_features ./ std(standardized_features);

% Perform PCA
[coeff, score, ~, ~, explained] = pca(standardized_features, "Algorithm","eig");
disp(explained)

cumulative_explained = cumsum(explained);
num_components = find(cumulative_explained >= 95, 1);
fprintf('Number of components to retain 95%% variance: %d\n', num_components);

% Replace original features with reduced components
reduced_features = score(:, 1:num_components);

% Split reduced features back into patient datasets
split_index = 0;
for patient = 1:10
    num_samples = size(all_data{patient}, 1);
    all_data{patient} = [reduced_features(split_index + 1:split_index + num_samples, :), all_data{patient}(:, 35)];
    split_index = split_index + num_samples;
end

% Prepare the metrics.
accuracies = zeros(1, 10);
sensitivities = zeros(1,10);
specificities = zeros(1,10);
precisions = zeros(1,10);
AUROCs = zeros(1,10);


% Perform leave-one-out cross-validation
for leave_out = 1:10
    test_features = all_data{leave_out}(:, 1:num_components); 
    test_labels = all_data{leave_out}(:, end); 

    train_features = [];
    train_labels = [];
    for train_patient = setdiff(1:10, leave_out)
        train_features = [train_features; all_data{train_patient}(:, 1:num_components)];
        train_labels = [train_labels; all_data{train_patient}(:, end)];
    end

    % Three different ML model implementations, the kNN application is
    % uncommented, but the rest can be uncommented and executed as well.

    % We tried fitting the data with a classification support vector machine model, using a linear kernel
    % function.

    % model = fitcsvm(train_features, train_labels, 'KernelFunction', 'linear');
    
    % Decision Tree implementation.
    % model = fitctree(train_features, train_labels); 

    % kNN implementation, the parameters were gained after the
    % "optimizehyperparameters" analysis.
    model = fitcknn(train_features, train_labels, "NumNeighbors",14, "Distance","cityblock", "Standardize",true);

    % Test the model on the left-out subject's data
    predictions = predict(model, test_features);

    accuracies(leave_out) = mean(predictions == test_labels);

    confMat = confusionmat(test_labels, predictions);
    % We assume that "positive" means "asleep" and "negative" means "awake"
    TP = confMat(1, 1); FN = confMat(1, 2);
    FP = confMat(2, 1); TN = confMat(2, 2);
    
    precisions(leave_out) = TP / (TP + FP);
    sensitivities(leave_out) = TP / (TP + FN);
    specificities(leave_out) = TN / (TN + FP);
    [~, ~, ~, AUROCs(leave_out)] = perfcurve(test_labels, predictions, 1); 
    
    if leave_out == 1 % just a single confusion matrix to not overwhelm the program with 10 figures. 
        figure;
        confusionchart(confMat, {'Asleep', 'Awake'});
    end

    fprintf('Processed patient %d/%d\n', leave_out, 10);
end

fprintf('Accuracies for each patient:\n');
disp(accuracies);

overall_accuracy = mean(accuracies);
fprintf('Overall accuracy: %.2f%%\n', overall_accuracy * 100);

fprintf('Precisions for each patient:\n');
disp(precisions);

overall_precision = mean(precisions);
fprintf('Overall precision: %.2f%%\n', overall_precision * 100);

fprintf('Sensitivities for each patient:\n');
disp(sensitivities);

overall_sensitivity = mean(sensitivities);
fprintf('Overall sensitivity: %.2f%%\n', overall_sensitivity * 100);

fprintf('Specificities for each patient:\n');
disp(specificities);

overall_specificity = mean(specificities);
fprintf('Overall specificity: %.2f%%\n', overall_specificity * 100);

fprintf('Area under the ROC curve for each patient:\n');
disp(AUROCs);

overall_AUROC = mean(AUROCs);
fprintf('Overall area under the ROC curve: %.2f%\n', overall_AUROC);





