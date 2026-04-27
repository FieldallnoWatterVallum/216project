function audio = record_audio(fs, duration)
    fprintf('录音 %.1f 秒...\n', duration);
    
    recorder = audiorecorder(fs, 16, 1);
    
    record(recorder);
    
    for t = duration:-0.5:0.5
        fprintf('%.1f... ', t);
        pause(0.5);
    end
    fprintf('\n');
    
    stop(recorder);
    
    audio = getaudiodata(recorder);
    
    if max(abs(audio)) > 0
        audio = audio / max(abs(audio));
    end
    
    fprintf('录音完成\n');
end