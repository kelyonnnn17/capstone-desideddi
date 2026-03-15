document.addEventListener('DOMContentLoaded', () => {
    const drug1Select = document.getElementById('drug1');
    const drug2Select = document.getElementById('drug2');
    const predictBtn = document.getElementById('predict-btn');
    const resultsContainer = document.getElementById('results-panel');
    const cardsGrid = document.querySelector('.cards-grid');
    const resultsMeta = document.getElementById('results-meta');
    const btnText = document.querySelector('.btn-text');
    const spinner = document.querySelector('.spinner');

    let isPredicting = false;

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

    drug1Select.addEventListener('change', validateSelection);
    drug2Select.addEventListener('change', validateSelection);

    // Handle Prediction Click
    predictBtn.addEventListener('click', () => {
        isPredicting = true;
        predictBtn.disabled = true;
        btnText.textContent = "Synthesizing...";
        spinner.classList.remove('hidden');
        resultsContainer.classList.add('hidden');
        cardsGrid.innerHTML = '';
        
        const payload = {
            drug1: parseInt(drug1Select.value),
            drug2: parseInt(drug2Select.value)
        };

        const d1Name = drug1Select.options[drug1Select.selectedIndex].text.split(' (')[0];
        const d2Name = drug2Select.options[drug2Select.selectedIndex].text.split(' (')[0];

        fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
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

            renderResults(data, d1Name, d2Name);
        })
        .catch(err => {
            console.error(err);
            isPredicting = false;
            btnText.textContent = "Compute Interaction Vector";
            spinner.classList.add('hidden');
            validateSelection();
            alert("Network error. Make sure the Flask backend is running.");
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
});
