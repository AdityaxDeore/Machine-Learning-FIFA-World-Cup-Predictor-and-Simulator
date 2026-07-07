// Global data and references
let allTeams = [];
let wcTeams = [];
let outcomeChartInstance = null;
let scorelinesChartInstance = null;
let winnerChartInstance = null;

const FLAG_MAPPINGS = {
    "England": "gb-eng", "Scotland": "gb-sct", "Wales": "gb-wls", "Northern Ireland": "gb-nir",
    "South Korea": "kr", "Korea Republic": "kr", "Iran": "ir", "USA": "us", "United States": "us",
    "Russia": "ru", "Ivory Coast": "ci", "Côte d'Ivoire": "ci", "Türkiye": "tr", "Turkey": "tr",
    "Czechia": "cz", "Czech Republic": "cz", "Bosnia and Herzegovina": "ba", "Bosnia & Herzegovina": "ba",
    "Democratic Republic of the Congo": "cd", "DR Congo": "cd", "Cape Verde": "cv", "Curaçao": "cw",
    "Haiti": "ht", "South Africa": "za", "New Zealand": "nz", "Saudi Arabia": "sa", "Egypt": "eg",
    "Algeria": "dz", "Jordan": "jo", "Uzbekistan": "uz", "Colombia": "co", "Croatia": "hr",
    "Ghana": "gh", "Panama": "pa", "Argentina": "ar", "Australia": "au", "Austria": "at",
    "Belgium": "be", "Brazil": "br", "Canada": "ca", "Ecuador": "ec", "France": "fr",
    "Germany": "de", "Japan": "jp", "Mexico": "mx", "Morocco": "ma", "Netherlands": "nl",
    "Norway": "no", "Paraguay": "py", "Portugal": "pt", "Qatar": "qa", "Senegal": "sn",
    "Spain": "es", "Sweden": "se", "Switzerland": "ch", "Tunisia": "tn", "Uruguay": "uy"
};

function getFlagUrl(teamName) {
    if (FLAG_MAPPINGS[teamName]) {
        return `https://flagcdn.com/w80/${FLAG_MAPPINGS[teamName]}.png`;
    }
    return `https://flagcdn.com/w80/un.png`;
}

// Initialize on DOM Load
document.addEventListener("DOMContentLoaded", () => {
    initTabs();
    loadTeamData();
    initPredictor();
    initSimulator();
    initStandings();
});

// 1. Tab Navigation System
function initTabs() {
    const tabs = document.querySelectorAll(".nav-tab");
    tabs.forEach(tab => {
        tab.addEventListener("click", () => {
            tabs.forEach(t => t.classList.remove("active"));
            tab.classList.add("active");
            
            const targetTab = tab.getAttribute("data-tab");
            document.querySelectorAll(".tab-content").forEach(content => {
                content.classList.remove("active");
            });
            document.getElementById(targetTab).classList.add("active");
        });
    });
}

// Load team list from FastAPI backend
async function loadTeamData() {
    try {
        const res = await fetch("/api/teams");
        const data = await res.json();
        allTeams = data.all_teams;
        wcTeams = data.wc_teams;
        
        populateTeamSelects("wc");
    } catch (e) {
        console.error("Error loading team list:", e);
    }
}

// Populate team dropdown selectors
function populateTeamSelects(poolType) {
    const select1 = document.getElementById("select-team1");
    const select2 = document.getElementById("select-team2");
    
    const teams = (poolType === "wc") ? wcTeams : allTeams;
    
    select1.innerHTML = "";
    select2.innerHTML = "";
    
    teams.forEach(team => {
        const opt1 = document.createElement("option");
        opt1.value = team;
        opt1.textContent = team;
        select1.appendChild(opt1);
        
        const opt2 = document.createElement("option");
        opt2.value = team;
        opt2.textContent = team;
        select2.appendChild(opt2);
    });
    
    // Set default selections (Portugal vs Spain)
    if (teams.includes("Portugal")) select1.value = "Portugal";
    if (teams.includes("Spain")) select2.value = "Spain";
    
    updatePreviews();
}

// 2. Predictor Logic
function initPredictor() {
    // Radio buttons pool change handler
    const poolRadios = document.querySelectorAll('input[name="team-pool"]');
    poolRadios.forEach(radio => {
        radio.addEventListener("change", (e) => {
            populateTeamSelects(e.target.value);
        });
    });
    
    // Select selectors handlers
    document.getElementById("select-team1").addEventListener("change", updatePreviews);
    document.getElementById("select-team2").addEventListener("change", updatePreviews);
    
    // Predict button click handler
    document.getElementById("btn-predict").addEventListener("click", runPrediction);
}

function updatePreviews() {
    const team1 = document.getElementById("select-team1").value;
    const team2 = document.getElementById("select-team2").value;
    
    document.getElementById("preview-name1").textContent = team1;
    document.getElementById("preview-name2").textContent = team2;
    
    document.getElementById("preview-flag1").src = getFlagUrl(team1);
    document.getElementById("preview-flag2").src = getFlagUrl(team2);
}

async function runPrediction() {
    const team1 = document.getElementById("select-team1").value;
    const team2 = document.getElementById("select-team2").value;
    const neutral = document.getElementById("neutral-venue").checked;

    if (team1 === team2) {
        alert("Please select two different teams!");
        return;
    }

    const btn = document.getElementById("btn-predict");
    btn.disabled = true;
    btn.textContent = "⚡ Running ML Model...";

    try {
        const response = await fetch("/api/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ team1, team2, neutral })
        });

        const data = await response.json();

        // Show results container
        document.getElementById("prediction-results").classList.remove("hidden");

        // --- Score forecast card ---
        document.getElementById("result-name1").textContent = team1;
        document.getElementById("result-name2").textContent = team2;
        document.getElementById("result-flag1").src = getFlagUrl(team1);
        document.getElementById("result-flag2").src = getFlagUrl(team2);
        document.getElementById("predicted-score1").textContent = data.predicted_score[0];
        document.getElementById("predicted-score2").textContent = data.predicted_score[1];

        // Update preview ELOs
        document.getElementById("preview-elo1").textContent = `ELO: ${data.team1_elo}`;
        document.getElementById("preview-elo2").textContent = `ELO: ${data.team2_elo}`;

        // --- Probability bars ---
        const pH = data.prob_home;
        const pD = data.prob_draw;
        const pA = data.prob_away;

        const pHp = (pH * 100).toFixed(1);
        const pDp = (pD * 100).toFixed(1);
        const pAp = (pA * 100).toFixed(1);

        document.getElementById("prob-label-home").textContent = team1;
        document.getElementById("prob-label-away").textContent = team2;

        // Animate bar widths
        setTimeout(() => {
            document.getElementById("prob-bar-home").style.width = `${pHp}%`;
            document.getElementById("prob-bar-draw").style.width = `${pDp}%`;
            document.getElementById("prob-bar-away").style.width = `${pAp}%`;
        }, 50);

        document.getElementById("prob-val-home").textContent = `${pHp}%`;
        document.getElementById("prob-val-draw").textContent = `DRAW: ${pDp}%`;
        document.getElementById("prob-val-away").textContent = `${pAp}%`;

        // --- Charts ---
        renderOutcomeChart(team1, team2, pH, pD, pA);
        renderScorelinesChart(data.scoreline_probs);

        // --- Detailed statistics grid ---
        const statsGrid = document.getElementById("stats-grid");
        statsGrid.innerHTML = `
            <div class="stat-row-comparison">
                <div class="stat-value left">${data.exp_g1.toFixed(2)} xG</div>
                <div class="stat-label">Expected Goals (xG)</div>
                <div class="stat-value right">${data.exp_g2.toFixed(2)} xG</div>
            </div>
            <div class="stat-row-comparison">
                <div class="stat-value left">${data.team1_elo}</div>
                <div class="stat-label">ELO Rating</div>
                <div class="stat-value right">${data.team2_elo}</div>
            </div>
            <div class="stat-row-comparison">
                <div class="stat-value left">${pHp}%</div>
                <div class="stat-label">Win Probability</div>
                <div class="stat-value right">${pAp}%</div>
            </div>
            <div class="stat-row-comparison">
                <div class="stat-value left">${data.predicted_score[0]} - ${data.predicted_score[1]}</div>
                <div class="stat-label">Predicted Score</div>
                <div class="stat-value right">${data.predicted_score[1]} - ${data.predicted_score[0]}</div>
            </div>
        `;

        if (data.metadata && data.metadata.team1_rank) {
            statsGrid.innerHTML += `
                <div class="stat-row-comparison">
                    <div class="stat-value left">#${data.metadata.team1_rank}</div>
                    <div class="stat-label">FIFA Rank</div>
                    <div class="stat-value right">#${data.metadata.team2_rank}</div>
                </div>
                <div class="stat-row-comparison">
                    <div class="stat-value left">${data.metadata.team1_manager || 'N/A'}</div>
                    <div class="stat-label">Manager</div>
                    <div class="stat-value right">${data.metadata.team2_manager || 'N/A'}</div>
                </div>
            `;
        }

        // --- Top Scorers ---
        const list1 = document.getElementById("scorers-list1");
        const list2 = document.getElementById("scorers-list2");
        document.getElementById("scorers-t1-name").textContent = team1;
        document.getElementById("scorers-t2-name").textContent = team2;

        list1.innerHTML = "";
        list2.innerHTML = "";

        const renderScorers = (list, scorers) => {
            if (!scorers || scorers.length === 0) {
                list.innerHTML = `<p class="section-desc">Player statistics not available.</p>`;
                return;
            }
            scorers.forEach(s => {
                const card = document.createElement("div");
                card.className = "player-card";
                card.innerHTML = `
                    <div class="player-name-box">
                        <span class="player-name">${s.name}</span>
                        <span class="player-pos">${s.position}</span>
                    </div>
                    <span class="player-prob">${(s.prob * 100).toFixed(1)}%</span>
                `;
                list.appendChild(card);
            });
        };
        renderScorers(list1, data.scorers1);
        renderScorers(list2, data.scorers2);

        // Scroll results into view smoothly
        document.getElementById("prediction-results").scrollIntoView({ behavior: "smooth", block: "start" });

    } catch (err) {
        console.error("Match prediction failed:", err);
        alert("Error running prediction model. Check backend server logs.");
    } finally {
        btn.disabled = false;
        btn.textContent = "⚡ Run ML Prediction";
    }
}


// Chart.js Win/Draw/Loss probability setup
function renderOutcomeChart(t1, t2, pH, pD, pA) {
    const ctx = document.getElementById("chart-outcome").getContext("2d");
    if (outcomeChartInstance) {
        outcomeChartInstance.destroy();
    }
    
    outcomeChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: [t1, 'Draw', t2],
            datasets: [{
                data: [(pH * 100).toFixed(1), (pD * 100).toFixed(1), (pA * 100).toFixed(1)],
                backgroundColor: ['#FF6B6B', '#C4B5FD', '#FFD93D'],
                borderColor: '#000000',
                borderWidth: 3,
                borderRadius: 0
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { 
                    grid: { display: false }, 
                    border: { color: '#000000', width: 3 },
                    ticks: { color: '#000000', font: { family: 'Space Grotesk', weight: 'bold' } } 
                },
                y: { 
                    grid: { display: false }, 
                    border: { color: '#000000', width: 3 },
                    ticks: { color: '#000000', font: { family: 'Space Grotesk', weight: '900' } } 
                }
            }
        }
    });
}

// Chart.js scoreline chart setup
function renderScorelinesChart(scorelines) {
    const ctx = document.getElementById("chart-scorelines").getContext("2d");
    if (scorelinesChartInstance) {
        scorelinesChartInstance.destroy();
    }
    
    scorelinesChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: scorelines.map(s => s.scoreline),
            datasets: [{
                data: scorelines.map(s => (s.prob * 100).toFixed(1)),
                backgroundColor: '#FFD93D',
                borderColor: '#000000',
                borderWidth: 3,
                borderRadius: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { 
                    grid: { display: false }, 
                    border: { color: '#000000', width: 3 },
                    ticks: { color: '#000000', font: { family: 'Space Grotesk', weight: '900' } } 
                },
                y: { 
                    grid: { display: false }, 
                    border: { color: '#000000', width: 3 },
                    ticks: { color: '#000000', font: { family: 'Space Grotesk', weight: 'bold' } } 
                }
            }
        }
    });
}

// 3. Knockout Simulator Logic
function initSimulator() {
    document.getElementById("btn-run-bracket").addEventListener("click", simulateSingleBracket);
    document.getElementById("btn-run-monte-carlo").addEventListener("click", runMonteCarloSimulation);
    
    // Sync slider display value (register once here, not inside render)
    const slider = document.getElementById("slider-runs");
    slider.addEventListener("input", (e) => {
        document.getElementById("slider-runs-val").textContent = e.target.value;
    });
}

async function simulateSingleBracket() {
    // Hide Monte Carlo results
    document.getElementById("monte-carlo-results").classList.add("hidden");
    
    const btn = document.getElementById("btn-run-bracket");
    btn.disabled = true;
    btn.textContent = "Simulating Bracket...";
    
    try {
        const response = await fetch("/api/simulate-bracket", { method: "POST" });
        const data = await response.json();
        
        document.getElementById("bracket-results").classList.remove("hidden");
        
        const container = document.getElementById("bracket-tree-container");
        container.innerHTML = "";
        
        // Helper: create matchup HTML node
        const createMatchBox = (m) => {
            const isWinnerT1 = m.winner === m.t1;
            const isWinnerT2 = m.winner === m.t2;
            const p1 = m.pen === m.t1 ? " (pen)" : "";
            const p2 = m.pen === m.t2 ? " (pen)" : "";
            
            return `
                <div class="bracket-match">
                    <div class="bracket-team ${isWinnerT1 ? 'winner' : ''}">
                        <div style="display:flex; align-items:center;">
                            <img src="${getFlagUrl(m.t1)}" class="bracket-team-flag" alt="${m.t1}">
                            <span class="bracket-team-name">${m.t1}</span>
                        </div>
                        <span class="bracket-team-score">${m.g1}${p1}</span>
                    </div>
                    <div class="bracket-team ${isWinnerT2 ? 'winner' : ''}">
                        <div style="display:flex; align-items:center;">
                            <img src="${getFlagUrl(m.t2)}" class="bracket-team-flag" alt="${m.t2}">
                            <span class="bracket-team-name">${m.t2}</span>
                        </div>
                        <span class="bracket-team-score">${m.g2}${p2}</span>
                    </div>
                </div>
            `;
        };
        
        // R16 Round Output
        let r16Html = `<div class="bracket-round-title">Round of 16</div>` + data.r16.map(createMatchBox).join("");
        // QF
        let qfHtml = `<div class="bracket-round-title">Quarterfinals</div>` + data.qf.map(createMatchBox).join("");
        // SF
        let sfHtml = `<div class="bracket-round-title">Semifinals</div>` + data.sf.map(createMatchBox).join("");
        // Final
        let finalHtml = `<div class="bracket-round-title">Final</div>` + createMatchBox(data.final);
        
        // Champion card
        let champHtml = `
            <div class="bracket-champ-card">
                <span class="bracket-champ-title">🏆 Champion</span>
                <img src="${getFlagUrl(data.champion)}" class="bracket-champ-flag" alt="${data.champion}">
                <div class="bracket-champ-name">${data.champion}</div>
            </div>
        `;
        
        // Columns assembly
        container.innerHTML = `
            <div class="bracket-column">${r16Html}</div>
            <div class="bracket-column">${qfHtml}</div>
            <div class="bracket-column">${sfHtml}</div>
            <div class="bracket-column">${finalHtml}</div>
            <div class="bracket-column" style="justify-content:center;">${champHtml}</div>
        `;
        
    } catch (e) {
        console.error("Bracket simulation failed:", e);
    } finally {
        btn.disabled = false;
        btn.textContent = "Simulate Single Run Bracket";
    }
}

async function runMonteCarloSimulation() {
    // Hide single bracket
    document.getElementById("bracket-results").classList.add("hidden");
    
    const btn = document.getElementById("btn-run-monte-carlo");
    btn.disabled = true;
    btn.textContent = "Simulating MC runs...";
    
    const runs = parseInt(document.getElementById("slider-runs").value);
    
    // Simulate progress bar
    const progressBox = document.getElementById("mc-progress-box");
    const progressFill = document.getElementById("mc-progress-fill");
    const progressText = document.getElementById("mc-progress-text");
    
    progressBox.classList.remove("hidden");
    document.getElementById("monte-carlo-results").classList.remove("hidden");
    
    let progress = 0;
    const interval = setInterval(() => {
        progress += 7;
        if (progress > 85) progress = 85;
        progressFill.style.width = `${progress}%`;
        progressText.textContent = `Running ${runs} simulations (${progress}%)...`;
    }, 150);
    
    try {
        const response = await fetch("/api/simulate-monte-carlo", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ runs })
        });
        
        const data = await response.json();
        
        clearInterval(interval);
        progressFill.style.width = `100%`;
        progressText.textContent = "Simulation complete!";
        
        setTimeout(() => {
            progressBox.classList.add("hidden");
        }, 1000);
        
        // Render Odds Data Table
        const tbody = document.querySelector("#table-monte-carlo tbody");
        tbody.innerHTML = "";
        
        data.forEach(row => {
            const tr = document.createElement("tr");
            tr.innerHTML = `
                <td><strong>${row.team}</strong></td>
                <td>${row.R16.toFixed(1)}%</td>
                <td>${row.QF.toFixed(1)}%</td>
                <td>${row.SF.toFixed(1)}%</td>
                <td>${row.Final.toFixed(1)}%</td>
                <td><strong style="color:var(--accent-color);">${row.Winner.toFixed(1)}%</strong></td>
            `;
            tbody.appendChild(tr);
        });
        
        // Draw winning probabilities chart
        renderWinnerChart(data.slice(0, 10)); // Top 10 teams
        
    } catch (e) {
        clearInterval(interval);
        progressBox.classList.add("hidden");
        console.error("Monte Carlo simulation failed:", e);
    } finally {
        btn.disabled = false;
        btn.textContent = "Run Multi-Run Simulation";
    }
}

function renderWinnerChart(mcData) {
    const ctx = document.getElementById("chart-winner").getContext("2d");
    if (winnerChartInstance) {
        winnerChartInstance.destroy();
    }
    
    winnerChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: mcData.map(r => r.team),
            datasets: [{
                data: mcData.map(r => r.Winner),
                backgroundColor: '#C4B5FD',
                borderColor: '#000000',
                borderWidth: 3,
                borderRadius: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { 
                    grid: { display: false }, 
                    border: { color: '#000000', width: 3 },
                    ticks: { color: '#000000', font: { family: 'Space Grotesk', weight: '900' } } 
                },
                y: { 
                    grid: { display: false }, 
                    border: { color: '#000000', width: 3 },
                    ticks: { color: '#000000', font: { family: 'Space Grotesk', weight: 'bold' } } 
                }
            }
        }
    });
    
}

// 4. ELO Standings Logic
function initStandings() {
    const grpFlt = document.getElementById("filter-group");
    const confFlt = document.getElementById("filter-conf");
    
    grpFlt.addEventListener("change", fetchStandings);
    confFlt.addEventListener("change", fetchStandings);
    
    fetchStandings();
}

async function fetchStandings() {
    const group = document.getElementById("filter-group").value;
    const confederation = document.getElementById("filter-conf").value;
    
    try {
        const queryParams = new URLSearchParams();
        if (group && group !== "All Groups") queryParams.append("group", group);
        if (confederation && confederation !== "All Confederations") queryParams.append("confederation", confederation);
        
        const res = await fetch(`/api/standings?${queryParams.toString()}`);
        const data = await res.json();
        
        const tbody = document.querySelector("#table-standings-list tbody");
        tbody.innerHTML = "";
        
        if (data.length === 0) {
            tbody.innerHTML = `<tr><td colspan="9" style="text-align:center;">No matching records found.</td></tr>`;
            return;
        }
        
        data.forEach((row, idx) => {
            const tr = document.createElement("tr");
            tr.innerHTML = `
                <td>${idx + 1}</td>
                <td><strong>${row.team}</strong></td>
                <td>Group ${row.group}</td>
                <td>${row.confederation}</td>
                <td>${row.manager}</td>
                <td>#${row.fifa_rank}</td>
                <td>${row.base_elo}</td>
                <td>${row.current_elo}</td>
                <td><strong style="color:${row.gain_loss.startsWith('+') ? 'var(--accent-color)' : '#ef4444'}">${row.gain_loss}</strong></td>
            `;
            tbody.appendChild(tr);
        });
    } catch (e) {
        console.error("Failed to load standings table:", e);
    }
}
