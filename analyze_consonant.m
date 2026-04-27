function analyze_consonant(segment, fs, consonant_name)
    
    figure('Name', sprintf('consonant "%s" analysis', consonant_name), ...
           'Position', [100, 100, 1000, 800]);
    
    subplot(3, 2, 1);
    t = (0:length(segment)-1)/fs * 1000; 
    plot(t, segment);
    xlabel('time (ms)');
    ylabel('amplitude');
    title('time domain waveform');
    grid on;
    xlim([0, t(end)]);
    
    subplot(3, 2, 2);
    NFFT = 1024;
    [Pxx, F] = pwelch(segment, hamming(256), 128, NFFT, fs);
    plot(F, 10*log10(Pxx));
    xlabel('frequency (Hz)');
    ylabel('power spectral density (dB/Hz)');
    title('power spectrum');
    grid on;
    xlim([0, 4000]);
    
    subplot(3, 2, 3);
    spectrogram(segment, hamming(256), 128, NFFT, fs, 'yaxis');
    title('spectrogram');
    ylim([0, 4]);  % 0-4kHz
    colorbar;
    
    subplot(3, 2, 4);
    [Pxx, F] = pwelch(segment, hamming(256), 128, NFFT, fs);
    semilogy(F, Pxx);
    xlabel('frequency (Hz)');
    ylabel('power');
    title('Spectral envelope (logarithmic coordinate)');
    grid on;
    xlim([0, 4000]);
    
    subplot(3, 2, 5);
    frame_len = round(0.01 * fs);  % 10ms frame
    num_frames = floor(length(segment) / frame_len);
    zcr = zeros(1, num_frames);
    
    for i = 1:num_frames
        start_idx = (i-1) * frame_len + 1;
        end_idx = min(start_idx + frame_len - 1, length(segment));
        frame = segment(start_idx:end_idx);
        zcr(i) = sum(abs(diff(sign(frame)))) / (2 * length(frame));
    end
    
    plot((1:num_frames) * frame_len/fs * 1000, zcr);
    xlabel('time (ms)');
    ylabel('zero-crossing rate');
    title('change in the zero-crossing rate');
    grid on;
    
    subplot(3, 2, 6);
    energy = zeros(1, num_frames);
    
    for i = 1:num_frames
        start_idx = (i-1) * frame_len + 1;
        end_idx = min(start_idx + frame_len - 1, length(segment));
        frame = segment(start_idx:end_idx);
        energy(i) = sum(frame.^2);
    end
    
    plot((1:num_frames) * frame_len/fs * 1000, energy);
    xlabel('time (ms)');
    ylabel('energy');
    title('short-time energy');
    grid on;
    
    sgtitle(sprintf('voiced consonant "%s" spectral analysis', consonant_name), ...
           'FontSize', 14, 'FontWeight', 'bold');
end