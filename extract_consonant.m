function segment = extract_consonant(audio, fs)
    
    % pre-empasis
    pre_emphasis = 0.97;
    audio = filter([1, -pre_emphasis], 1, audio);
    
    % take pre-100ms
    segment_length = round(0.1 * fs);  % 100ms
    segment_length = min(segment_length, length(audio));
    
    segment = audio(1:segment_length);
    
    % add window
    window = hamming(length(segment));
    segment = segment .* window;
end