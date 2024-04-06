%Recommended to use .wav files but .mp3 will work too

[noisy_audio, fs] = audioread('Hallelujah.wav');

% Convert to mono if necessary
if size(noisy_audio, 2) > 1
    noisy_audio = mean(noisy_audio, 2);
end

% Pre-processing: High-pass filtering
cutoff_frequency = 200;  % Adjust based on speech characteristics
[b, a] = butter(4, cutoff_frequency/(fs/2), 'high');
filtered_audio = filter(b, a, noisy_audio);

% Spectral analysis: STFT
window_size = 256;  % Adjust based on desired time-frequency resolution
overlap = round(0.5 * window_size);  % Convert overlap fraction to integer
[filtered_spectrum, f, t] = spectrogram(filtered_audio, window_size, overlap, fs);

% Noise reduction: Spectral subtraction with noise estimation
estimated_noise_spectrum = estimate_noise_spectrum(filtered_spectrum);
enhanced_spectrum = subtract_noise(filtered_spectrum, estimated_noise_spectrum);

% Post-processing: Inverse STFT and spectral tilt compensation (optional)
enhanced_audio = inverse_stft(enhanced_spectrum, window_size, overlap);

% Ensure both signals have the same length
min_length = min(length(noisy_audio), length(enhanced_audio));
noisy_audio = noisy_audio(1:min_length);
enhanced_audio = enhanced_audio(1:min_length);

% Analysis and evaluation
% Calculate SNR
enhanced_SNR = snr(noisy_audio, enhanced_audio);
fprintf('Enhanced SNR: %.2f dB\n', enhanced_SNR);

% Plot time-domain signals and spectrograms
figure;

% Plot original and filtered signals
subplot(3, 1, 1);
plot(noisy_audio, 'b');
title('Original Speech Signal');

subplot(3, 1, 2);
plot(filtered_audio, 'g');
title('Filtered Speech Signal');

% Plot enhanced signal and its spectrogram
subplot(3, 1, 3);
plot(enhanced_audio, 'r');
title('Enhanced Speech Signal');

% Plot time-domain signals and spectrograms
figure;
subplot(2, 1, 1);
plot(noisy_audio, 'b');
hold on;
plot(enhanced_audio, 'r');
title('Original (blue) vs Enhanced (red) Speech Signal');
legend('Original', 'Enhanced');

subplot(2, 1, 2);
imagesc(t, f, abs(enhanced_spectrum));
title('Spectrogram of Enhanced Speech');

% Subjective listening
soundsc([noisy_audio, enhanced_audio]);  % Play original and enhanced speech

function noise_spectrum = estimate_noise_spectrum(filtered_spectrum)
    % Estimate noise spectrum by averaging across frequency bins
    noise_spectrum = mean(filtered_spectrum, 2);
end

function enhanced_spectrum = subtract_noise(filtered_spectrum, noise_spectrum)
    % Perform spectral subtraction
    alpha = 0.0001;  % Adjust based on noise characteristics
    enhanced_spectrum = filtered_spectrum - alpha * noise_spectrum;
    enhanced_spectrum(enhanced_spectrum < 0) = 0;  % Thresholding to avoid negative values
end

function enhanced_audio = inverse_stft(spectrum, window_size, overlap)
    % Perform inverse STFT
    [rows, cols] = size(spectrum);
    signal_length = (cols - 1) * overlap + window_size;
    enhanced_audio = zeros(signal_length, 1);
    
    for col = 1:cols
        start_idx = (col - 1) * overlap + 1;
        end_idx = start_idx + window_size - 1;
        enhanced_audio(start_idx:end_idx) = enhanced_audio(start_idx:end_idx) + ifft(spectrum(:, col), window_size);
    end
    
    enhanced_audio = real(enhanced_audio);
end

