% The code that we used to extract features from signals.
fs = 100; % Sampling frequency

for patient = 1:10
    % Load EEG data
    eeg_filename = sprintf("patient%02d-PSG.edf", patient);
    eeg_data = edfread(eeg_filename);
    EEGFpz_Cz_vectors = eeg_data.EEGFpz_Cz; % Extract EEG data, specifically, the Fpz-Cz electrode data.
    % Change EEGFpz_Cz into "Pz_Oz" to extract the signals from the other
    % channel.

    % Load Hypnograms
    hypnogram_filename = sprintf("patient%02d-Hypnogram.edf", patient);
    [~, eeg_labels_timetable] = edfread(hypnogram_filename);
    
    % Code for labelling the segments, as the hypnogram timetable does not
    % label every segment.
    onset_times = seconds(eeg_labels_timetable.Onset);
    eeg_labels = eeg_labels_timetable.Annotations; 

    num_segments = length(EEGFpz_Cz_vectors);
    num_vectors = num_segments;
    fprintf('Patient %d - Number of segments: %d\n', patient, num_segments);  
    segment_labels = strings(num_segments, 1);

    for i = 1:num_segments
        
        current_time = (i - 1) * 30; 
        matched_label = "NaN";

        for onset_idx = 1:length(onset_times) - 1
            if onset_times(onset_idx) <= current_time && onset_times(onset_idx + 1) > current_time
                matched_label = eeg_labels{onset_idx};
                break;
            end
        end
        segment_labels(i) = matched_label;
    end

    % Initialize feature matrix
    feature_matrix = zeros(num_vectors, 35);

    % We used the first signal of each patient to calculate the Itakura
    % distance. The first vector of all 10 patients are classified as
    % "awake", so they can be used as a baseline signal.

    baseline_signal = EEGFpz_Cz_vectors{1}; 
    p = 10; % An AR order of 10, a tentative selection.

    % Compute autocorrelation and AR coefficients for the baseline
    r_baseline = xcorr(baseline_signal, p, 'biased');
    r_baseline = r_baseline(p+1:end);
    [baseline_ar_coeffs, ~] = levinson(r_baseline, p); % AR coefficients are calculated using the Levinson method, as proposed by researchers.
    [H, R_baseline] = corrmtx(baseline_signal, p); % the correlation matrix of the baseline signal.

    % We had to use the transposed version at the end instead of the
    % beginning (the formula that we encountered used the beginning),
    % because the matrix multiplication was not compatible otherwise.
    MSE_xx = baseline_ar_coeffs * R_baseline * baseline_ar_coeffs'; 

    % Iterate through each EEG vector
    for i = 1:num_vectors
        current_signal = EEGFpz_Cz_vectors{i};

        % Compute autocorrelation and AR coefficients
        r_current = xcorr(current_signal, p, 'biased');
        r_current = r_current(p+1:end);
        [current_ar_coeffs, ~] = levinson(r_current, p);

        % Compute Itakura distance
        MSE_xy = current_ar_coeffs * R_baseline * current_ar_coeffs';
        itakura_distance = log(MSE_xy / MSE_xx);

        % Extract statistical features
        min_val = min(current_signal);
        max_val = max(current_signal);
        mean_val = mean(current_signal);
        std_val = std(current_signal);
        skewness_val = skewness(current_signal);
        kurtosis_val = kurtosis(current_signal);

        % Compute first and second derivatives
        first_derivative = diff(current_signal);
        max_first_derivative = max(first_derivative);
        mean_abs_first_derivative = mean(abs(first_derivative));

        second_derivative = diff(current_signal, 2);
        max_second_derivative = max(second_derivative);
        mean_abs_second_derivative = mean(abs(second_derivative));

        % Compute zero crossing rate
        zero_crossing_rate = zerocrossrate(current_signal);

        % Compute entropy
        signal_prob = histcounts(current_signal, 'Normalization', 'probability');
        entropy_val = -sum(signal_prob .* log2(signal_prob + eps));

        % Compute line length and RMS
        line_length = sum(abs(diff(current_signal)));
        rms_val = sqrt(mean(current_signal.^2));

        % Compute nonlinear energy
        nonlinear_energy = sum(current_signal(2:end-1).^2 - current_signal(1:end-2) .* current_signal(3:end));

        % Compute mobility and complexity
        mobility = std(first_derivative) / std_val;
        complexity = std(second_derivative) / std(first_derivative) / mobility;

        % Compute spectral features
        nfft = 512;
        [pxx, f] = pwelch(current_signal, [], [], nfft, fs); % Computed the power using Welch's spectral density method.

        % Used the band frequencies used by the researchers.
        delta_band = (f >= 0.5 & f <= 4);
        theta_band = (f >= 4 & f <= 8);
        alpha_band = (f >= 8 & f <= 12);
        beta_band = (f >= 12 & f <= 30);

        delta_power = sum(pxx(delta_band));
        theta_power = sum(pxx(theta_band));
        alpha_power = sum(pxx(alpha_band));
        beta_power = sum(pxx(beta_band));

        total_power = sum(pxx);
        delta_relative = delta_power / total_power;
        theta_relative = theta_power / total_power;
        alpha_relative = alpha_power / total_power;
        beta_relative = beta_power / total_power;

        % Wavelet decomposition, db4 was selected as the mother wavelet.
        [c, l] = wavedec(current_signal, 4, 'db4');

        % The coefficients were extracted by following the lecture slides.
        delta_coeffs = appcoef(c, l, 'db4', 4);
        theta_coeffs = detcoef(c, l, 4);
        alpha_coeffs = detcoef(c, l, 3);
        beta_coeffs = detcoef(c, l, 2);

        delta_energy = sum(delta_coeffs.^2);
        theta_energy = sum(theta_coeffs.^2);
        alpha_energy = sum(alpha_coeffs.^2);
        beta_energy = sum(beta_coeffs.^2);

        total_energy = delta_energy + theta_energy + alpha_energy + beta_energy;

        % All the extracted features are then added to the feature matrix.
        feature_matrix(i, :) = [
            min_val, max_val, mean_val, std_val, skewness_val, kurtosis_val, ...
            max_first_derivative, mean_abs_first_derivative, ...
            max_second_derivative, mean_abs_second_derivative, ...
            zero_crossing_rate, entropy_val, itakura_distance, ...
            line_length, rms_val, nonlinear_energy, ...
            mobility, complexity, delta_power, theta_power, alpha_power, beta_power, ...
            delta_relative, theta_relative, alpha_relative, beta_relative, ...
            delta_energy, theta_energy, alpha_energy, beta_energy, ...
            (delta_energy / total_energy) * 100, ...
            (theta_energy / total_energy) * 100, ...
            (alpha_energy / total_energy) * 100, ...
            (beta_energy / total_energy) * 100, ...
            double(segment_labels(i) == "Sleep stage W") % 1 if awake, 0 if sleeping.
        ];
    end

    % We then saved the results of each patient within these files.
    results_filename = sprintf('Patient_%02d_FeatureMatrix.mat', patient);
    save(results_filename, 'feature_matrix');
end

