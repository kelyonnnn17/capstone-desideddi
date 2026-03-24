document.addEventListener('DOMContentLoaded', () => {
    const drug1Select = document.getElementById('drug1');
    const drug2Select = document.getElementById('drug2');
    const predictBtn = document.getElementById('predict-btn');
    const resultsContainer = document.getElementById('results-panel');
    const cardsGrid = document.querySelector('.cards-grid');
    const resultsMeta = document.getElementById('results-meta');
    const btnText = document.querySelector('.btn-text');
    const spinner = document.querySelector('.spinner');
    
    // LLM Export Variables
    const exportLlmBtn = document.getElementById('export-llm-btn');
    const exportBtnText = exportLlmBtn ? exportLlmBtn.querySelector('.btn-text') : null;
    const exportSpinner = exportLlmBtn ? exportLlmBtn.querySelector('.spinner') : null;
    const analyzeLlmBtn = document.getElementById('analyze-llm-btn');
    const analyzeBtnText = analyzeLlmBtn ? analyzeLlmBtn.querySelector('.btn-text') : null;
    const analyzeSpinner = analyzeLlmBtn ? analyzeLlmBtn.querySelector('.spinner') : null;
    const llmAnalysisPanel = document.getElementById('llm-analysis-panel');
    const llmAnalysisMeta = document.getElementById('llm-analysis-meta');
    const llmAnalysisOutput = document.getElementById('llm-analysis-output');

    let isPredicting = false;
    let isExporting = false;
    let isAnalyzing = false;
    let lastLlmPayload = null;
    let lastPredictionKey = null;
    let lastAnalysis = null;
    let lastTopResults = [];

    // Load available drugs on page load
    fetch('/api/metadata')
        .then(response => response.json())
        .then(data => {
            if(data.status === 'success') {
                populateDropdowns(data.drugs);
            } else {
                console.error("Failed to load metadata");
            }
        })
        .catch(err => {
            console.error('API Error:', err);
            drug1Select.innerHTML = `<option value="">Error Loading Database</option>`;
            drug2Select.innerHTML = `<option value="">Error Loading Database</option>`;
        });

    function populateDropdowns(drugs) {
        // Sort drugs alphabetically 
        drugs.sort((a,b) => a.name.localeCompare(b.name));
        
        let htmlOptions = `<option value="" disabled selected>Select a Molecule...</option>`;
        drugs.forEach(d => {
            htmlOptions += `<option value="${d.pubchemID}">${d.name} (CID: ${d.pubchemID})</option>`;
        });

        drug1Select.innerHTML = htmlOptions;
        drug2Select.innerHTML = htmlOptions;
        
        drug1Select.disabled = false;
        drug2Select.disabled = false;
    }

    // Enable predict button only when both drugs are selected AND they are different
    function validateSelection() {
        const d1 = drug1Select.value;
        const d2 = drug2Select.value;
        
        if (d1 && d2 && d1 !== d2) {
            predictBtn.disabled = isPredicting;
        } else {
            predictBtn.disabled = true;
        }
    }

    function currentPredictionKey() {
        return `${drug1Select.value || ''}:${drug2Select.value || ''}`;
    }

    function canSubmitPrediction() {
        return drug1Select.value && drug2Select.value && drug1Select.value !== drug2Select.value && !isPredicting;
    }

    function setExportReady(isReady) {
        if (!exportLlmBtn) return;
        exportLlmBtn.disabled = !isReady || isExporting;
    }

    function setAnalyzeReady(isReady) {
        if (!analyzeLlmBtn) return;
        analyzeLlmBtn.disabled = !isReady || isAnalyzing || isPredicting;
    }

    function clearAnalysis() {
        lastAnalysis = null;
        if (llmAnalysisPanel) llmAnalysisPanel.classList.add('hidden');
        if (llmAnalysisMeta) llmAnalysisMeta.textContent = '';
        if (llmAnalysisOutput) llmAnalysisOutput.textContent = '';
    }

    drug1Select.addEventListener('change', validateSelection);
    drug2Select.addEventListener('change', validateSelection);
    drug1Select.addEventListener('change', () => {
        lastLlmPayload = null;
        lastPredictionKey = null;
        lastTopResults = [];
        setExportReady(false);
        setAnalyzeReady(false);
        clearAnalysis();
    });
    drug2Select.addEventListener('change', () => {
        lastLlmPayload = null;
        lastPredictionKey = null;
        lastTopResults = [];
        setExportReady(false);
        setAnalyzeReady(false);
        clearAnalysis();
    });

    function triggerPrediction() {
        if (!canSubmitPrediction()) return;

        isPredicting = true;
        predictBtn.disabled = true;
        btnText.textContent = "Running DeSIDE + BioMistral...";
        spinner.classList.remove('hidden');
        resultsContainer.classList.add('hidden');
        cardsGrid.innerHTML = '';
        setExportReady(false);
        setAnalyzeReady(false);
        clearAnalysis();
        
        const payload = {
            drug1: parseInt(drug1Select.value),
            drug2: parseInt(drug2Select.value)
        };

        const d1Name = drug1Select.options[drug1Select.selectedIndex].text.split(' (')[0];
        const d2Name = drug2Select.options[drug2Select.selectedIndex].text.split(' (')[0];

        fetch('/api/analyze_llm', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                ...payload,
                include_payload: false,
                include_prompt: false
            })
        })
        .then(res => res.json())
        .then(data => {
            isPredicting = false;
            btnText.textContent = "Compute Interaction Vector";
            spinner.classList.add('hidden');
            validateSelection();
            
            if(data.error) {
                alert(`Prediction Error: ${data.error}`);
                return;
            }

            lastLlmPayload = null;
            lastPredictionKey = currentPredictionKey();
            lastTopResults = data.top_results || [];
            lastAnalysis = data.analysis || data.simple_analysis || '';
            setExportReady(true);
            setAnalyzeReady(true);
            renderResults(data, d1Name, d2Name);
            renderAnalysis(data);
            loadGraph(payload.drug1, payload.drug2);
        })
        .catch(err => {
            console.error(err);
            isPredicting = false;
            btnText.textContent = "Compute Interaction Vector";
            spinner.classList.add('hidden');
            validateSelection();
            setExportReady(false);
            setAnalyzeReady(false);
            alert("Network error. Make sure the Flask backend is running.");
        });
    }

    // Handle Prediction Click
    predictBtn.addEventListener('click', triggerPrediction);
    [drug1Select, drug2Select].forEach(select => {
        select.addEventListener('keydown', event => {
            if (event.key === 'Enter') {
                event.preventDefault();
                triggerPrediction();
            }
        });
    });

    function renderResults(data, d1Name, d2Name) {
        resultsContainer.classList.remove('hidden');
        
        const count = data.total_predicted_reactions;
        
        if (count === 0) {
            resultsMeta.innerHTML = `Model predicts <strong>0</strong> significant side effect interactions between <strong>${d1Name}</strong> and <strong>${d2Name}</strong>. Safe to co-administer based on current latent parameters.`;
            return;
        }

        resultsMeta.innerHTML = `Found potentially <strong>${count}</strong> adverse reactions between <strong>${d1Name}</strong> and <strong>${d2Name}</strong>. Showing top flagged risks:`;

        let cardsHtml = '';
        
        data.top_results.forEach(se => {
            // Determine severity colors based on generic confidence gaps
            let severityClass = 'card-med';
            let severityLabel = 'Moderate Risk';
            
            if(se.confidence_gap > 0.15) {
                severityClass = 'card-severe';
                severityLabel = 'Critical Risk';
            } else if(se.confidence_gap > 0.08) {
                severityClass = 'card-high';
                severityLabel = 'High Risk';
            }

            cardsHtml += `
                <div class="se-card ${severityClass}">
                    <div class="se-name">${se.name}</div>
                    <div style="margin-bottom: 15px; font-size: 0.8rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; color: var(--text-secondary);">
                        ${severityLabel}
                    </div>
                    <div class="se-stats">
                        <span>Raw Score:</span>
                        <span class="stat-val">${(se.probability_score).toFixed(2)}</span>
                    </div>
                    <div class="se-stats" style="margin-top: 8px;">
                        <span>Danger Gap:</span>
                        <span class="stat-val">+${(se.confidence_gap).toFixed(3)}</span>
                    </div>
                </div>
            `;
        });

        cardsGrid.innerHTML = cardsHtml;
    }

    function renderAnalysis(data) {
        if (!llmAnalysisPanel || !llmAnalysisOutput || !llmAnalysisMeta) return;
        llmAnalysisPanel.classList.remove('hidden');
        llmAnalysisMeta.textContent = `Model: ${data.ollama?.model || 'unknown'} | Reviewed side effect: ${data.selected_side_effect || 'N/A'}${data.ollama?.repaired ? ' | Output repaired to required format' : ''}`;
        llmAnalysisOutput.textContent = data.analysis || data.simple_analysis || 'No analysis returned.';
    }

    // Handle LLM Export Click
    if(exportLlmBtn) {
        setExportReady(false);
        exportLlmBtn.addEventListener('click', () => {
            if (isExporting || isPredicting) return;
            
            isExporting = true;
            exportLlmBtn.disabled = true;
            exportBtnText.textContent = "Extracting Embeddings...";
            exportSpinner.classList.remove('hidden');

            const payload = {
                drug1: parseInt(drug1Select.value),
                drug2: parseInt(drug2Select.value)
            };

            const exportPromise = fetch('/api/export_llm', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(payload)
                }).then(res => res.json());

            exportPromise
            .then(data => {
                isExporting = false;
                exportBtnText.textContent = "Export LLM Payload JSON";
                exportSpinner.classList.add('hidden');
                setExportReady(true);
                
                if(data.error) {
                    alert(`Export Error: ${data.error}`);
                    return;
                }

                lastLlmPayload = data.llm_payload || null;
                lastPredictionKey = currentPredictionKey();

                const blob = new Blob([JSON.stringify(data.llm_payload, null, 2)], { type: 'application/json' });
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `DeSIDE_LLM_Payload_D${payload.drug1}_D${payload.drug2}.json`;
                document.body.appendChild(a);
                a.click();
                
                // Cleanup
                window.URL.revokeObjectURL(url);
                document.body.removeChild(a);
            })
            .catch(err => {
                console.error("Export Error: ", err);
                isExporting = false;
                exportBtnText.textContent = "Export LLM Payload JSON";
                exportSpinner.classList.add('hidden');
                setExportReady(Boolean(lastLlmPayload));
                alert('Failed to extract LLM payload from backend engine.');
            });
        });
    }

    if(analyzeLlmBtn) {
        setAnalyzeReady(false);
        analyzeLlmBtn.addEventListener('click', () => {
            if (isAnalyzing || isPredicting) return;

            isAnalyzing = true;
            analyzeLlmBtn.disabled = true;
            analyzeBtnText.textContent = "Running BioMistral...";
            analyzeSpinner.classList.remove('hidden');

            const topEffect = lastTopResults?.[0]?.name || null;
            const payload = {
                drug1: parseInt(drug1Select.value),
                drug2: parseInt(drug2Select.value),
                predicted_side_effect: topEffect,
                include_payload: false,
                include_prompt: false
            };

            fetch('/api/analyze_llm', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(payload)
            })
            .then(res => res.json())
            .then(data => {
                isAnalyzing = false;
                analyzeBtnText.textContent = "Analyze with BioMistral";
                analyzeSpinner.classList.add('hidden');
                setAnalyzeReady(Boolean(lastPredictionKey));

                if (data.error) {
                    alert(`LLM Analysis Error: ${data.error}`);
                    return;
                }

                lastTopResults = data.top_results || lastTopResults;
                lastAnalysis = data.analysis || data.simple_analysis || '';
                renderAnalysis(data);
            })
            .catch(err => {
                console.error("LLM Analysis Error:", err);
                isAnalyzing = false;
                analyzeBtnText.textContent = "Analyze with BioMistral";
                analyzeSpinner.classList.add('hidden');
                setAnalyzeReady(Boolean(lastPredictionKey));
                alert('Failed to run the Ollama BioMistral analysis.');
            });
        });
    }
});
