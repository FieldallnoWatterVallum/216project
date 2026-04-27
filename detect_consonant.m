function [detected, score, detected_consonant, all_scores] = detect_consonant(audio, fs, templates, target_consonant)
    test_segment = extract_consonant(audio, fs);
    test_features = extract_features(test_segment, fs);
    consonant_names = fieldnames(templates);
    scores = zeros(length(consonant_names), 1);
    
    for i = 1:length(consonant_names)
        consonant = consonant_names{i};
        template_features = templates.(consonant);
        [similarity, details] = compute_similarity_improved(test_features, template_features);
        scores(i) = similarity;
        
        fprintf('%s: %.4f\n', consonant, similarity);
    end

    [best_score, best_idx] = max(scores);
    detected_consonant = consonant_names{best_idx};

    threshold = 0.6; 
    detected = (strcmp(detected_consonant, target_consonant) && best_score >= threshold);
    score = best_score;
    all_scores = scores; 
    fprintf('%s, %.4f\n', detected_consonant, best_score);
    fprintf('%.2f, %s\n', threshold, string(detected));
end

function [similarity, details] = compute_similarity_improved(feat1, feat2)
    weights = struct(...
        'mfcc', 0.40, ...      
        'mfcc_delta', 0.10, ...
        'spectral', 0.15, ...  
        'formants', 0.10, ...
        'temporal', 0.15, ...   
        'pitch', 0.10);
    
    details = struct();
    
    if isfield(feat1, 'mfcc_mean') && isfield(feat2, 'mfcc_mean')
        mfcc1 = feat1.mfcc_mean;
        mfcc2 = feat2.mfcc_mean;
        
        details.mfcc = cosine_similarity_normalized(mfcc1, mfcc2);
        
        if isfield(feat1, 'mfcc_delta_mean') && isfield(feat2, 'mfcc_delta_mean')
            delta1 = feat1.mfcc_delta_mean;
            delta2 = feat2.mfcc_delta_mean;
            details.mfcc_delta = cosine_similarity_normalized(delta1, delta2);
        else
            details.mfcc_delta = 0.5;
        end
    else
        details.mfcc = 0.5;
        details.mfcc_delta = 0.5;
    end
    
    spectral_feat_names = {'spectral_centroid', 'spectral_rolloff85', 'spectral_flatness'};
    spectral_sims = zeros(length(spectral_feat_names), 1);
    
    for i = 1:length(spectral_feat_names)
        name = spectral_feat_names{i};
        if isfield(feat1, name) && isfield(feat2, name)
            val1 = feat1.(name);
            val2 = feat2.(name);
            
            if isinf(val1) || isinf(val2) || isnan(val1) || isnan(val2) || ...
               val1 <= 0 || val2 <= 0
                spectral_sims(i) = 0.5;
                continue;
            end
            
            log_val1 = log10(val1 + eps);
            log_val2 = log10(val2 + eps);
            diff_ratio = abs(log_val1 - log_val2) / (abs(log_val2) + 1);
            spectral_sims(i) = exp(-diff_ratio);
        else
            spectral_sims(i) = 0.5;
        end
    end
    details.spectral = mean(spectral_sims);
    
    if isfield(feat1, 'formants') && isfield(feat2, 'formants')
        formant1 = feat1.formants;
        formant2 = feat2.formants;
        
        valid_idx = (formant2 > 0) & (formant1 > 0);
        if any(valid_idx)
            formant_sims = zeros(1, sum(valid_idx));
            for i = 1:length(formant_sims)
                f1 = formant1(valid_idx(i));
                f2 = formant2(valid_idx(i));
                rel_diff = abs(f1 - f2) / ((f1 + f2)/2 + eps);
                formant_sims(i) = exp(-rel_diff / 0.5);  % 50%
            end
            details.formants = mean(formant_sims);
        else
            details.formants = 0.5;
        end
    else
        details.formants = 0.5;
    end
    
    temporal_feat_names = {'zcr', 'energy', 'rms'};
    temporal_sims = zeros(length(temporal_feat_names), 1);
    
    for i = 1:length(temporal_feat_names)
        name = temporal_feat_names{i};
        if isfield(feat1, name) && isfield(feat2, name)
            val1 = feat1.(name);
            val2 = feat2.(name);
            
            if isinf(val1) || isinf(val2) || isnan(val1) || isnan(val2)
                temporal_sims(i) = 0.5;
                continue;
            end
            
            if strcmp(name, 'energy') || strcmp(name, 'rms')
                if val1 > eps && val2 > eps
                    log_val1 = log10(val1);
                    log_val2 = log10(val2);
                    diff_ratio = abs(log_val1 - log_val2) / (abs(log_val2) + 1);
                    temporal_sims(i) = exp(-diff_ratio);
                else
                    temporal_sims(i) = 0.5;
                end
            else
                if val2 > 0
                    rel_diff = abs(val1 - val2) / (val2 + eps);
                    temporal_sims(i) = exp(-rel_diff / 0.5);
                else
                    temporal_sims(i) = 0.5;
                end
            end
        else
            temporal_sims(i) = 0.5;
        end
    end
    details.temporal = mean(temporal_sims);
    
    if isfield(feat1, 'f0') && isfield(feat2, 'f0')
        f0_1 = feat1.f0;
        f0_2 = feat2.f0;
        
        if f0_1 > 0 && f0_2 > 0
            rel_diff = abs(f0_1 - f0_2) / ((f0_1 + f0_2)/2 + eps);
            details.pitch = exp(-rel_diff / 0.3);  
        else
            details.pitch = 0.5;
        end
    else
        details.pitch = 0.5;
    end
    
    similarity = weights.mfcc * details.mfcc + ...
                 weights.mfcc_delta * details.mfcc_delta + ...
                 weights.spectral * details.spectral + ...
                 weights.formants * details.formants + ...
                 weights.temporal * details.temporal + ...
                 weights.pitch * details.pitch;
    
    similarity = max(0, min(1, similarity));
end

function sim = cosine_similarity_normalized(vec1, vec2)
    vec1_norm = (vec1 - min(vec1)) / (max(vec1) - min(vec1) + eps);
    vec2_norm = (vec2 - min(vec2)) / (max(vec2) - min(vec2) + eps);
    
    sim = dot(vec1_norm, vec2_norm) / (norm(vec1_norm) * norm(vec2_norm) + eps);
    sim = (sim + 1) / 2; 
end