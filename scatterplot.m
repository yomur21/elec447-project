% A small piece of code that we used to obtain the scatter plot in our
% presentation.
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
    
% Use the first 2 principal components and plot a 2D scatter plot of all
% data points for the first 4 patients. 
    figure()
    sp(1) = subplot(2,2,1);
    gscatter(score(1:size(all_data{1}, 1),1), score(1:size(all_data{1}, 1),2), all_data{1}(:, end))
    title("Patient 1")
    legend('Sleeping', 'Awake')
    sp(2) = subplot(2,2,2);
    gscatter(score(1:size(all_data{2}, 1),1), score(1:size(all_data{2}, 1),2), all_data{2}(:, end))
    title("Patient 2")
    legend('Sleeping', 'Awake')
    sp(3) = subplot(2,2,3);
    gscatter(score(1:size(all_data{3}, 1),1), score(1:size(all_data{3}, 1),2), all_data{3}(:, end))
    title("Patient 3")
    legend('Sleeping', 'Awake')
    sp(4) = subplot(2,2,4);
    gscatter(score(1:size(all_data{4}, 1),1), score(1:size(all_data{4}, 1),2), all_data{4}(:, end))
    title("Patient 4")
    legend('Sleeping', 'Awake')

