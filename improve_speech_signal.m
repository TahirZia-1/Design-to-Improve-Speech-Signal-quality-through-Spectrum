%% Speech Signal Enhancement
% MP3 files will work as well

function enhance_speech(input_file, output_file)
    % Input validation
    if nargin < 1
        input_file = 'Hallelujah.wav';
    end
    if nargin < 2
        [filepath, name, ~] = fileparts(input_file);
        output_file = fullfile(filepath, [name '_enhanced.wav']);
    end
    
    % Display processing message
    fprintf('Processing file: %s\n', input_file);
    
    %% Load audio file
    try
        [noisy_audio, fs] = audioread(input_file);
        fprintf('File loaded successfully. Sample rate: %d Hz\n', fs);
    catch ME
        error('Error loading audio file: %s', ME.message);
    end

    %% Convert to mono if necessary
    if size(noisy_audio, 2) > 1
        noisy_audio = mean(noisy_audio, 2);
        fprintf('Converted stereo to mono\n');
    end
    
    % Normalize audio
    noisy_audio = noisy_audio / max(abs(noisy_audio));
    
    %% Pre-processing: High-pass filtering
    % Parameters can be adjusted based on the speech characteristics
    cutoff_frequency = 200;  % Hz
    filter_order = 4;
    
    % Design and apply Butterworth high-pass filter
    [b, a] = butter(filter_order, cutoff_frequency/(fs/2), 'high');
    filtered_audio = filter(b, a, noisy_audio);
    fprintf('Applied high-pass filter with cutoff frequency: %d Hz\n', cutoff_frequency);
    
    %% Spectral analysis: STFT
    % Parameters for time-frequency resolution balance
    window_size = 256;
    overlap_percentage = 0.75;  % Increased for better results
    overlap = round(overlap_percentage * window_size);
    nfft = 512;  % Increased FFT size for better frequency resolution
    
    % Perform STFT with Hamming window for better spectral estimation
    window = hamming(window_size);
    [filtered_spectrum, f, t] = spectrogram(filtered_audio, window, overlap, nfft, fs);
    fprintf('Performed STFT with window size: %d, overlap: %.2f%%\n', window_size, overlap_percentage*100);
    
    %% Noise reduction: Improved spectral subtraction with adaptive noise estimation
    % Estimate noise spectrum from beginning frames (assuming initial frames contain mostly noise)
    num_noise_frames = min(10, size(filtered_spectrum, 2));
    estimated_noise_spectrum = estimate_noise_spectrum(filtered_spectrum(:, 1:num_noise_frames));
    
    % Apply enhanced spectral subtraction with adaptive parameters
    enhanced_spectrum = adaptive_spectral_subtraction(filtered_spectrum, estimated_noise_spectrum);
    fprintf('Applied adaptive spectral subtraction for noise reduction\n');
    
    %% Post-processing: Inverse STFT and enhancement
    enhanced_audio = inverse_stft(enhanced_spectrum, window, overlap, nfft);
    
    % Apply de-emphasis to compensate for spectral tilt
    enhanced_audio = apply_deemphasis(enhanced_audio, 0.95);
    
    % Apply soft-thresholding for residual noise reduction
    threshold = 0.01;
    enhanced_audio = soft_threshold(enhanced_audio, threshold);
    
    % Ensure both signals have the same length
    min_length = min(length(noisy_audio), length(enhanced_audio));
    noisy_audio = noisy_audio(1:min_length);
    enhanced_audio = enhanced_audio(1:min_length);
    
    % Normalize enhanced audio
    enhanced_audio = enhanced_audio / max(abs(enhanced_audio));
    
    %% Save enhanced audio
    try
        audiowrite(output_file, enhanced_audio, fs);
        fprintf('Enhanced audio saved to: %s\n', output_file);
    catch ME
        warning('Error saving enhanced audio: %s', ME.message);
    end
    
    %% Analysis and evaluation
    % Calculate Signal-to-Noise Ratio (SNR)
    original_SNR = calculate_snr(noisy_audio);
    enhanced_SNR = calculate_snr(enhanced_audio);
    
    fprintf('Original SNR: %.2f dB\n', original_SNR);
    fprintf('Enhanced SNR: %.2f dB\n', enhanced_SNR);
    fprintf('SNR Improvement: %.2f dB\n', enhanced_SNR - original_SNR);
    
    % Calculate Perceptual Evaluation of Speech Quality (PESQ)-like metric
    pesq_score = simple_pesq(noisy_audio, enhanced_audio);
    fprintf('Speech quality score: %.2f (higher is better)\n', pesq_score);
    
    %% Visualization
    % Create figure with adjustable size
    figure('Name', 'Speech Enhancement Results', 'Position', [100, 100, 800, 600]);
    
    % Plot original and filtered signals
    subplot(3, 1, 1);
    plot((0:length(noisy_audio)-1)/fs, noisy_audio, 'b');
    title('Original Speech Signal');
    xlabel('Time (s)');
    ylabel('Amplitude');
    grid on;
    
    subplot(3, 1, 2);
    plot((0:length(filtered_audio)-1)/fs, filtered_audio, 'g');
    title('Filtered Speech Signal');
    xlabel('Time (s)');
    ylabel('Amplitude');
    grid on;
    
    % Plot enhanced signal
    subplot(3, 1, 3);
    plot((0:length(enhanced_audio)-1)/fs, enhanced_audio, 'r');
    title('Enhanced Speech Signal');
    xlabel('Time (s)');
    ylabel('Amplitude');
    grid on;
    
    % Create another figure for comparison and spectrogram
    figure('Name', 'Comparison and Spectrogram', 'Position', [100, 100, 800, 600]);
    
    subplot(2, 1, 1);
    plot((0:length(noisy_audio)-1)/fs, noisy_audio, 'b');
    hold on;
    plot((0:length(enhanced_audio)-1)/fs, enhanced_audio, 'r');
    title('Original (blue) vs Enhanced (red) Speech Signal');
    xlabel('Time (s)');
    ylabel('Amplitude');
    legend('Original', 'Enhanced');
    grid on;
    
    subplot(2, 1, 2);
    % Create enhanced spectrogram with better visual parameters
    spectrogram(enhanced_audio, window, overlap, nfft, fs, 'yaxis');
    title('Spectrogram of Enhanced Speech');
    colormap('jet');
    colorbar;
    
    % Play original and enhanced speech with a short pause between
    disp('Playing original audio followed by enhanced audio...');
    soundsc(noisy_audio, fs);
    pause(length(noisy_audio)/fs + 1); % Wait for original audio to finish + 1 second
    soundsc(enhanced_audio, fs);
end

%% Helper Functions

function noise_spectrum = estimate_noise_spectrum(noise_frames)
    % Improved noise spectrum estimation
    % Average magnitude spectrum across specified noise frames
    noise_spectrum = mean(abs(noise_frames), 2);
    
    % Apply smoothing to reduce spectral variance
    window_size = 5;
    if length(noise_spectrum) > window_size
        noise_spectrum = smooth(noise_spectrum, window_size);
    end
end

function enhanced_spectrum = adaptive_spectral_subtraction(input_spectrum, noise_spectrum)
    % Adaptive spectral subtraction with oversubtraction factor
    % Initialize enhanced spectrum
    [rows, cols] = size(input_spectrum);
    enhanced_spectrum = zeros(rows, cols);
    
    % Oversubtraction factor (varies with frequency)
    alpha = 4.0;
    beta = 0.002;  % Spectral floor parameter
    
    % Process each time frame
    for col = 1:cols
        % Get current magnitude spectrum
        mag_spectrum = abs(input_spectrum(:, col));
        phase_spectrum = angle(input_spectrum(:, col));
        
        % Calculate local SNR
        snr_local = mag_spectrum ./ (noise_spectrum + eps);
        
        % Adjust oversubtraction factor based on SNR
        alpha_local = alpha ./ (1 + snr_local.^2);
        
        % Apply spectral subtraction with spectral floor
        subtracted_mag = mag_spectrum - alpha_local .* noise_spectrum;
        
        % Apply spectral floor
        floor_mag = beta * mag_spectrum;
        subtracted_mag = max(subtracted_mag, floor_mag);
        
        % Reconstruct complex spectrum
        enhanced_spectrum(:, col) = subtracted_mag .* exp(1i * phase_spectrum);
    end
end

function enhanced_audio = inverse_stft(spectrum, window, overlap, nfft)
    % Perform inverse STFT with improved overlap-add method
    [~, cols] = size(spectrum);
    hop_size = length(window) - overlap;
    
    % Calculate output signal length
    output_length = (cols - 1) * hop_size + length(window);
    enhanced_audio = zeros(output_length, 1);
    
    % Initialize normalization buffer
    norm_buffer = zeros(output_length, 1);
    
    % Overlap-add reconstruction
    for col = 1:cols
        % Inverse FFT
        current_frame = real(ifft(spectrum(:, col), nfft));
        
        % Extract time-domain frame (truncate if needed)
        current_frame = current_frame(1:length(window));
        
        % Apply synthesis window
        windowed_frame = current_frame .* window;
        
        % Calculate start and end indices
        start_idx = (col - 1) * hop_size + 1;
        end_idx = start_idx + length(window) - 1;
        end_idx = min(end_idx, output_length);
        
        % Add to output signal
        frame_length = end_idx - start_idx + 1;
        enhanced_audio(start_idx:end_idx) = enhanced_audio(start_idx:end_idx) + windowed_frame(1:frame_length);
        
        % Update normalization buffer
        norm_buffer(start_idx:end_idx) = norm_buffer(start_idx:end_idx) + window(1:frame_length).^2;
    end
    
    % Normalize output signal
    idx = norm_buffer > 1e-10;
    enhanced_audio(idx) = enhanced_audio(idx) ./ norm_buffer(idx);
end

function output = apply_deemphasis(input, coefficient)
    % Apply de-emphasis filter to compensate for spectral tilt
    % y(n) = x(n) + coefficient * y(n-1)
    output = filter(1, [1, -coefficient], input);
end

function output = soft_threshold(input, threshold)
    % Apply soft thresholding for residual noise reduction
    output = zeros(size(input));
    
    % For values above threshold, subtract threshold
    idx = input > threshold;
    output(idx) = input(idx) - threshold;
    
    % For values below negative threshold, add threshold
    idx = input < -threshold;
    output(idx) = input(idx) + threshold;
end

function snr_value = calculate_snr(signal)
    % Estimate SNR based on signal statistics
    % Assumes noise is present in low-energy segments
    
    % Segment the signal
    frame_length = 256;
    num_frames = floor(length(signal) / frame_length);
    
    % Calculate energy of each frame
    frame_energy = zeros(num_frames, 1);
    for i = 1:num_frames
        start_idx = (i - 1) * frame_length + 1;
        end_idx = start_idx + frame_length - 1;
        frame = signal(start_idx:end_idx);
        frame_energy(i) = sum(frame.^2);
    end
    
    % Sort frame energies
    sorted_energy = sort(frame_energy);
    
    % Use the lowest 10% frames as noise estimate
    noise_frames = max(1, round(0.1 * num_frames));
    noise_energy = mean(sorted_energy(1:noise_frames));
    
    % Use the highest 80% frames as signal+noise estimate
    signal_start = round(0.2 * num_frames) + 1;
    signal_noise_energy = mean(sorted_energy(signal_start:end));
    
    % Calculate SNR
    if noise_energy > 0
        signal_energy = signal_noise_energy - noise_energy;
        snr_value = 10 * log10(signal_energy / noise_energy);
    else
        snr_value = 100; % Very high SNR if no noise detected
    end
end

function score = simple_pesq(reference, processed)
    % A simple function that approximates speech quality
    % Not the actual PESQ algorithm but provides a relative quality score
    
    % Normalize signals
    reference = reference / max(abs(reference));
    processed = processed / max(abs(processed));
    
    % Calculate correlation coefficient
    correlation = abs(xcorr(reference, processed, 0, 'coeff'));
    
    % Calculate log spectral distance
    n_fft = 512;
    ref_spec = abs(fft(reference, n_fft)).^2;
    proc_spec = abs(fft(processed, n_fft)).^2;
    
    % Avoid log of zero
    ref_spec = ref_spec + eps;
    proc_spec = proc_spec + eps;
    
    % Calculate log spectral distance
    log_spec_dist = mean(abs(10*log10(ref_spec) - 10*log10(proc_spec)));
    
    % Convert to a 1-5 scale (5 being best quality)
    log_spec_factor = max(0, min(1, 1 - log_spec_dist/50));
    
    % Combine metrics (simple weighted average)
    score = 1 + 4 * (0.7 * correlation + 0.3 * log_spec_factor);
end

% If this script is run directly
if ~isemked('enhance_speech')
    enhance_speech();
end


