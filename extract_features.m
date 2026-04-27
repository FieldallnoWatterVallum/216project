function features = extract_features(segment, fs)
    if size(segment, 2) > 1
        segment = segment(:);
    end
    
    pre_emphasis = 0.97;
    segment = filter([1, -pre_emphasis], 1, segment);
    
    features = struct();
    
    features.mean = mean(segment);
    features.std = std(segment);
    
    features.zcr = sum(abs(diff(sign(segment)))) / (2 * length(segment));
    
    features.energy = sum(segment.^2) / length(segment);
    
    features.rms = rms(segment);
    
    NFFT = 512;
    [Pxx, F] = pwelch(segment, hamming(NFFT), NFFT/2, NFFT, fs);
    
    if sum(Pxx) > 0
        features.spectral_centroid = sum(F .* Pxx) / sum(Pxx);
    else
        features.spectral_centroid = 0;
    end

    cum_power = cumsum(Pxx);
    total_power = sum(Pxx);
    idx_rolloff85 = find(cum_power >= 0.85 * total_power, 1);
    idx_rolloff50 = find(cum_power >= 0.50 * total_power, 1);
    
    if ~isempty(idx_rolloff85)
        features.spectral_rolloff85 = F(idx_rolloff85);
    else
        features.spectral_rolloff85 = 0;
    end
    
    if ~isempty(idx_rolloff50)
        features.spectral_rolloff50 = F(idx_rolloff50);
    else
        features.spectral_rolloff50 = 0;
    end
    
    geometric_mean = exp(mean(log(Pxx + eps)));
    arithmetic_mean = mean(Pxx);
    features.spectral_flatness = geometric_mean / arithmetic_mean;
    
    Pxx_norm = Pxx / sum(Pxx);
    Pxx_norm(Pxx_norm == 0) = [];
    features.spectral_entropy = -sum(Pxx_norm .* log2(Pxx_norm));

    features.formants = estimate_formants_corrected(segment, fs);
    
    try
        [coeffs, ~, ~] = mfcc(segment, fs, 'NumCoeffs', 13);
        features.mfcc_mean = mean(coeffs, 2);
        features.mfcc_std = std(coeffs, 0, 2);
        
        if size(coeffs, 2) > 2
            delta_coeffs = diff(coeffs, 1, 2);
            features.mfcc_delta_mean = mean(delta_coeffs, 2);
        else
            features.mfcc_delta_mean = zeros(13, 1);
        end
    catch
        features.mfcc_mean = zeros(13, 1);
        features.mfcc_std = zeros(13, 1);
        features.mfcc_delta_mean = zeros(13, 1);
    end

    [f0, voiced_ratio] = estimate_pitch_corrected(segment, fs);
    features.f0 = f0;
    features.voiced_ratio = voiced_ratio;
end

function formants = estimate_formants_corrected(segment, fs)
    pre_emphasis = 0.97;
    segment = filter([1, -pre_emphasis], 1, segment);
    
    order = round(2 + fs/1000);
    
    try
        a = lpc(segment, order);
        r = roots(a);
        
        r = r(imag(r) > 0);
        
        angles = atan2(imag(r), real(r));
        frequencies = angles * (fs / (2*pi));
        frequencies = frequencies(frequencies > 200 & frequencies < 4000);
        frequencies = sort(frequencies);

        if length(frequencies) >= 3
            formants = frequencies(1:3)';
        elseif length(frequencies) > 0
            formants = [frequencies', zeros(1, 3-length(frequencies))];
        else
            formants = zeros(1, 3);
        end
    catch
        formants = zeros(1, 3);
    end
end


function [f0, voiced_ratio] = estimate_pitch_corrected(audio, fs)  
    frame_len = round(0.025 * fs);  % 25ms
    hop_len = round(0.010 * fs);    % 10ms
    
    num_frames = floor((length(audio) - frame_len) / hop_len) + 1;
    f0_values = zeros(1, num_frames);
    voiced = false(1, num_frames);
    
    for i = 1:num_frames
        start_idx = (i-1) * hop_len + 1;
        end_idx = min(start_idx + frame_len - 1, length(audio));
        frame = audio(start_idx:end_idx);
        
        frame = frame - mean(frame);
        autocorr = xcorr(frame, 'coeff');
        autocorr = autocorr(length(frame):end);
        
        [peaks, locs] = findpeaks(autocorr);
        
        if ~isempty(peaks)
            [max_peak, max_idx] = max(peaks);
            lag = locs(max_idx);
      
            if max_peak > 0.3 && lag >= round(fs/400) && lag <= round(fs/80)
                f0_values(i) = fs / lag;
                voiced(i) = true;
            end
        end
    end
    
    valid_f0 = f0_values(voiced);
    if ~isempty(valid_f0)
        f0 = median(valid_f0);
    else
        f0 = 0;
    end
    
    voiced_ratio = sum(voiced) / num_frames;
end