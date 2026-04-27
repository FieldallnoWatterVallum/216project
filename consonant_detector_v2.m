clear; clc; close all;

fs = 16000;  
consonants = {'b', 'd', 'g', 'v'};

for i = 1:length(consonants)
    fprintf('  %d. %s_consonant_clean.wav \n', i, consonants{i}, consonants{i});
end
fprintf('\n');

templates = struct();  
missing_files = {};    

for i = 1:length(consonants)
    consonant = consonants{i};
    filename = sprintf('%s_consonant_clean.wav', consonant);
    
    fprintf('%s\n', consonant);
    fprintf('%s\n', filename);

    if exist(filename, 'file')
        [audio, fs_orig] = audioread(filename);
        
        if size(audio, 2) > 1
            audio = mean(audio, 2);
        end
        
        if fs_orig ~= fs
            audio = resample(audio, fs, fs_orig);
        end
        
        consonant_segment = extract_consonant(audio, fs);

        analyze_consonant(consonant_segment, fs, consonant);

        templates.(consonant) = extract_features(consonant_segment, fs);
        
        fprintf(' "%s" Train Completed\n\n', consonant);
    else
        fprintf(' File "%s" not exist！\n', filename);
        missing_files{end+1} = filename;
        response = input('', 's');
        if strcmpi(response, 'y')
            audio = record_audio(fs, 2.0);
            audiowrite(filename, audio, fs);
            fprintf('  Save to: %s\n', filename);
            
            consonant_segment = extract_consonant(audio, fs);
            templates.(consonant) = extract_features(consonant_segment, fs);
        else
            fprintf('  Skip "%s"\n', consonant);
        end
    end
end

trained_consonants = fieldnames(templates);
if length(trained_consonants) < 2
    for i = 1:length(consonants)
        fprintf('  %s_sample.wav\n', consonants{i});
    end
    return;
end

for i = 1:length(trained_consonants)
    fprintf('  %s\n', trained_consonants{i});
end
fprintf('\n');

test_files = {
    'dog.wav', 'd';    
    'go.wav', 'g';     
    'vase.wav', 'v';  
    'book.wav', 'b';   
    'door.wav', 'd';    
};

existing_tests = {};
for i = 1:size(test_files, 1)
    if exist(test_files{i, 1}, 'file')
        existing_tests{end+1, 1} = test_files{i, 1};
        existing_tests{end, 2} = test_files{i, 2};
    end
end

if isempty(existing_tests)
    for i = 1:size(test_files, 1)
        fprintf('  %s\n', test_files{i, 1});
    end
    
    create_default_test = input('\nCreate default autofile?(y/n): ', 's');
    if strcmpi(create_default_test, 'y')
        for i = 1:4
            pause;
            audio = record_audio(fs, 2.0);
            audiowrite(test_files{i, 1}, audio, fs);
            existing_tests{end+1, 1} = test_files{i, 1};
            existing_tests{end, 2} = test_files{i, 2};
        end
    else
        return;
    end
end

fprintf('\n');

results = cell(size(existing_tests, 1), 5);

for i = 1:size(existing_tests, 1)
    test_file = existing_tests{i, 1};
    target_consonant = existing_tests{i, 2};
    
    
    [test_audio, fs_test] = audioread(test_file);
    
    if size(test_audio, 2) > 1
        test_audio = mean(test_audio, 2);
    end

    if fs_test ~= fs
        test_audio = resample(test_audio, fs, fs_test);
    end
    
    [detected, score, detected_cons] = detect_consonant(test_audio, fs, templates, target_consonant);
    
    results{i, 1} = test_file;
    results{i, 2} = target_consonant;
    results{i, 3} = detected_cons;
    results{i, 4} = score;
    results{i, 5} = detected;
    
    if detected
        fprintf(' Expected "%s" (Confidence: %.3f)\n', target_consonant, score);
    else
        if strcmp(detected_cons, 'none')
            fprintf('  ✗\n');
        else
            fprintf('  ✗ Get "%s"，but expect to "%s" (置信度: %.3f)\n', ...
                detected_cons, target_consonant, score);
        end
    end
    
    play_audio = input('  Play the audio？(y/n, default n): ', 's');
    if strcmpi(play_audio, 'y')
        soundsc(test_audio, fs);
        pause(length(test_audio)/fs + 0.5);
    end
end

fprintf('%-20s %-10s %-10s %-10s %-10s\n', ...
    'Test file', 'Target consonant', 'Detect result', 'Confidence', 'Status');
fprintf('%s\n', repmat('-', 60, 1));

correct_count = 0;
for i = 1:size(results, 1)
    status = '✗';
    if results{i, 5}
        status = '✓';
        correct_count = correct_count + 1;
    end
    
    fprintf('%-20s %-10s %-10s %-10.3f %-10s\n', ...
        results{i, 1}, results{i, 2}, results{i, 3}, results{i, 4}, status);
end

fprintf('%s\n', repmat('-', 60, 1));
fprintf('Correct Rate: %.1f%% (%d/%d)\n\n', ...
    100 * correct_count / size(results, 1), correct_count, size(results, 1));


save_choice = input('\nSave as module？(y/n): ', 's');
if strcmpi(save_choice, 'y')
    save('consonant_templates.mat', 'templates', 'fs', 'consonants');
    fprintf('Module saved to consonant_templates.mat\n');
end