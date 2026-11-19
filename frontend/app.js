// SatyaCheck Indian Fake News Detector — Frontend Logic

let allFetchedArticles = [];
let currentFilter = 'all';

document.addEventListener("DOMContentLoaded", () => {
    checkHealth();
    setupLiveTextListener();
    setupSearchKeypress();
    // Automatically load live search on start
    executeNewsSearch();
});

// Setup Enter key press in the Omni Search bar
function setupSearchKeypress() {
    const searchInput = document.getElementById("live-search-input");
    if (searchInput) {
        searchInput.addEventListener("keydown", (e) => {
            if (e.key === "Enter") {
                e.preventDefault();
                executeNewsSearch();
            }
        });
    }
}

// Switch tabs between Real-Time Search Results and Deep Scanner
function switchTab(tabId) {
    const tabLive = document.getElementById("tab-live");
    const tabAnalyze = document.getElementById("tab-analyze");
    const secLive = document.getElementById("section-live");
    const secAnalyze = document.getElementById("section-analyze");

    if (tabId === "live") {
        tabLive.classList.add("active");
        tabAnalyze.classList.remove("active");
        secLive.classList.remove("hidden");
        secAnalyze.classList.add("hidden");
    } else {
        tabAnalyze.classList.add("active");
        tabLive.classList.remove("active");
        secAnalyze.classList.remove("hidden");
        secLive.classList.add("hidden");
    }
}

// Quick Trending Topic Search
function quickSearch(topic) {
    const input = document.getElementById("live-search-input");
    if (input) {
        input.value = topic;
        executeNewsSearch();
    }
}

// Execute News Search against Real-Time Google News RSS & AI Pipeline
async function executeNewsSearch() {
    const query = document.getElementById("live-search-input").value.trim() || "India Breaking News";
    const lang = document.getElementById("search-lang").value || "en";
    const limit = document.getElementById("search-limit").value || "15";

    const submitBtn = document.getElementById("search-submit-btn");
    const spinner = document.getElementById("search-spinner");
    const btnText = submitBtn.querySelector(".btn-text");
    const container = document.getElementById("live-feed-container");
    const heading = document.getElementById("current-search-heading");
    const badge = document.getElementById("results-count-badge");

    submitBtn.disabled = true;
    spinner.classList.remove("hidden");
    btnText.textContent = "Scanning Live Outlets...";

    heading.innerHTML = `Scanning Indian media for: <em>"${escapeHtml(query)}"</em> (${lang === 'hi' ? 'हिन्दी' : 'English'})`;
    container.innerHTML = `
        <div class="loading-state" style="text-align:center; padding: 40px;">
            <div class="spinner" style="margin: 0 auto 12px; width: 28px; height: 28px; border-width: 3px;"></div>
            <p style="color: var(--text-secondary); font-size: 0.95rem;">Retrieving articles from Indian publishers & running XLM-RoBERTa deception check...</p>
        </div>
    `;

    // Make sure we're on the Live Search tab
    switchTab("live");

    try {
        const url = `/api/search_news?query=${encodeURIComponent(query)}&limit=${limit}&lang=${lang}`;
        const res = await fetch(url);
        const data = await res.json();

        if (res.ok && data.articles && data.articles.length > 0) {
            allFetchedArticles = data.articles;
            badge.textContent = allFetchedArticles.length;
            document.getElementById("feed-last-updated").textContent = `Updated: ${new Date().toLocaleTimeString()}`;
            renderArticles(allFetchedArticles);
        } else {
            allFetchedArticles = [];
            badge.textContent = "0";
            container.innerHTML = `
                <div class="loading-state" style="text-align:center; padding: 40px; color: var(--text-muted);">
                    <p style="font-size: 1.1rem; color: #fff; margin-bottom: 6px;">No recent articles matched your search.</p>
                    <p>Try broader terms like <em>"Modi"</em>, <em>"Supreme Court"</em>, <em>"RBI"</em>, or click one of the trending chips above.</p>
                </div>
            `;
        }
    } catch (err) {
        container.innerHTML = `<div class="loading-state" style="color: var(--red); padding: 20px;">Connection failed: ${err.message}</div>`;
    } finally {
        submitBtn.disabled = false;
        spinner.classList.add("hidden");
        btnText.textContent = "🔍 Search & Verify";
    }
}

// Render Article Cards
function renderArticles(articles) {
    const container = document.getElementById("live-feed-container");

    let filtered = articles;
    if (currentFilter === 'fake') {
        filtered = articles.filter(a => a.analysis.verdict === 'FAKE');
    } else if (currentFilter === 'real') {
        filtered = articles.filter(a => a.analysis.verdict === 'REAL');
    }

    if (filtered.length === 0) {
        container.innerHTML = `<div class="loading-state" style="text-align:center; padding: 30px; color: var(--text-muted);">No articles in this filter category.</div>`;
        return;
    }

    container.innerHTML = filtered.map((art, idx) => {
        const a = art.analysis;
        const isFake = a.verdict === "FAKE";
        const fakeProb = parseFloat(a.fake_probability.replace('%', ''));

        return `
            <div class="feed-card">
                <div class="feed-info">
                    <h3>${escapeHtml(art.title)}</h3>
                    ${art.snippet ? `<p class="feed-snippet">${escapeHtml(art.snippet)}</p>` : ''}
                    <div class="feed-meta">
                        <span class="feed-source-chip">🏛️ ${escapeHtml(art.source || "Unknown Publisher")}</span>
                        <span>📅 ${escapeHtml(art.pub_date ? art.pub_date.substring(0, 16) : "Recent")}</span>
                        <span>🗣️ ${a.language}</span>
                        <span>🎭 ${a.sentiment}</span>
                    </div>
                </div>

                <div class="feed-card-actions">
                    <div class="feed-badge ${isFake ? 'fake' : 'real'}">
                        ${isFake ? '🔴 High Risk' : '🟢 Verified Likely'}<br>
                        <small>${a.fake_probability} Fake Prob</small>
                    </div>
                    <button class="deep-scan-link" onclick="loadInDeepScanner(${idx})">🔬 Deep Scan in Lab</button>
                    ${art.link ? `<a href="${art.link}" target="_blank" rel="noopener" class="source-external-link">Read Source Article ↗</a>` : ''}
                </div>
            </div>
        `;
    }).join("");
}

// Filter results (All / Fake / Real)
function filterResults(type) {
    currentFilter = type;
    document.querySelectorAll(".pill-filter").forEach(btn => btn.classList.remove("active"));

    if (type === 'all') document.querySelector(".pill-filter:nth-child(1)").classList.add("active");
    if (type === 'fake') document.querySelector(".fake-pill").classList.add("active");
    if (type === 'real') document.querySelector(".real-pill").classList.add("active");

    renderArticles(allFetchedArticles);
}

// Autofill into Deep Scanner
function loadInDeepScanner(idx) {
    const art = allFetchedArticles[idx];
    if (!art) return;

    document.getElementById("news-text").value = art.title;
    document.getElementById("source-input").value = art.source || "";

    switchTab("analyze");
    runAnalysis();
}

// Live language detection feedback as user types
function setupLiveTextListener() {
    const textarea = document.getElementById("news-text");
    const langTag = document.getElementById("live-lang-detect");

    textarea.addEventListener("input", () => {
        const text = textarea.value.trim();
        if (!text) {
            langTag.textContent = "Language: Auto-detecting...";
            return;
        }

        // Check Devanagari Unicode block
        const hasDevanagari = /[\u0900-\u097F]/.test(text);
        if (hasDevanagari) {
            langTag.textContent = "Detected: Hindi / Devanagari 🇮🇳";
            return;
        }

        const lower = text.toLowerCase();
        const hinglishTokens = ["modi", "sarkar", "yojana", "desh", "paisa", "bhai", "kya", "nahi", "shri", "bharat"];
        const matchCount = hinglishTokens.filter(w => lower.includes(w)).length;

        if (matchCount >= 2) {
            langTag.textContent = "Detected: Hinglish (Romanized Hindi) 🇮🇳";
        } else {
            langTag.textContent = "Detected: English (Indian Context) 🌐";
        }
    });
}

// Preset examples
function setExample(num) {
    const textarea = document.getElementById("news-text");
    const sourceInput = document.getElementById("source-input");

    if (num === 1) {
        textarea.value = "UNESCO declares Jana Gana Mana as best national anthem in emergency summit meeting.";
        sourceInput.value = "whatsapp forward";
    } else if (num === 2) {
        textarea.value = "आज रात 12 बजे से 500 रुपये के सभी नोट बंद हो जाएंगे, सरकार ने जारी किया आपातकालीन नोटिस।";
        sourceInput.value = "postcard.news";
    } else if (num === 3) {
        textarea.value = "ISRO successfully launches indigenous oceanographic satellite from Sriharikota space centre.";
        sourceInput.value = "thehindu.com";
    }

    textarea.dispatchEvent(new Event("input"));
}

function setSource(val) {
    document.getElementById("source-input").value = val;
}

// Check Backend status
async function checkHealth() {
    const statusChip = document.getElementById("model-status");
    try {
        const res = await fetch("/api/health");
        if (res.ok) {
            const data = await res.json();
            const modelTag = data.custom_trained ? "(Custom IFND)" : "(Base Weights)";
            statusChip.textContent = `Engine: ${data.model} ${modelTag}`;
        }
    } catch (e) {
        statusChip.textContent = "Backend: Connecting...";
    }
}

// Run Analysis via API
async function runAnalysis() {
    const text = document.getElementById("news-text").value.trim();
    const source = document.getElementById("source-input").value.trim();
    const btn = document.getElementById("analyze-btn");
    const spinner = document.getElementById("spinner");
    const btnText = btn.querySelector(".btn-text");

    if (!text) {
        alert("Please enter news headline or text to analyze.");
        return;
    }

    btn.disabled = true;
    spinner.classList.remove("hidden");
    btnText.textContent = "Scanning with XLM-RoBERTa...";

    try {
        const response = await fetch("/api/analyze", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text, source })
        });

        const data = await response.json();

        if (response.ok) {
            renderResults(data);
        } else {
            alert(data.error || "Failed to analyze text.");
        }
    } catch (err) {
        alert("Server connection error: " + err.message);
    } finally {
        btn.disabled = false;
        spinner.classList.add("hidden");
        btnText.textContent = "⚡ Scan for Deception";
    }
}

// Render Results & Animate Custom Gauge
function renderResults(data) {
    document.getElementById("res-empty").classList.add("hidden");
    const activeBox = document.getElementById("res-active");
    activeBox.classList.remove("hidden");

    // Timestamp
    document.getElementById("res-timestamp").textContent = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

    // Banner
    const banner = document.getElementById("verdict-banner");
    const pill = document.getElementById("verdict-pill");
    const confVal = document.getElementById("confidence-val");

    banner.className = "verdict-banner " + (data.verdict === "FAKE" ? "fake" : "real");
    pill.textContent = data.verdict === "FAKE" ? "🔴 FAKE NEWS" : "🟢 REAL NEWS";
    confVal.textContent = data.confidence;

    // Numerical values
    const fakeP = parseFloat(data.fake_probability.replace("%", ""));
    document.getElementById("gauge-num").textContent = `${fakeP.toFixed(1)}%`;

    // Animate SVG Gauge Arc
    // Full arc length for radius 80 semicircle = π * 80 ≈ 251.2
    const totalLength = 251.2;
    const offset = totalLength - (totalLength * (fakeP / 100));
    const arc = document.getElementById("gauge-arc");
    arc.style.strokeDashoffset = offset;

    // Stat cards
    document.getElementById("val-source-cred").textContent = `${data.source_credibility} (${data.source_score})`;
    document.getElementById("val-sentiment").textContent = data.sentiment;
    document.getElementById("val-language").textContent = data.language;

    // Risk level & hoax/forward alerts
    const riskEl = document.getElementById("val-risk-level");
    if (riskEl) {
        riskEl.textContent = data.risk_level || "Standard Text Analysis";
        riskEl.className = "stat-value";
        if (data.risk_level && data.risk_level.includes("Hoax")) riskEl.style.color = "#ef4444";
        else if (data.risk_level && data.risk_level.includes("Forward")) riskEl.style.color = "#f59e0b";
        else riskEl.style.color = "";
    }
    const hoaxEl = document.getElementById("val-hoax-flags");
    if (hoaxEl) {
        if (data.matched_hoaxes && data.matched_hoaxes.length > 0) {
            hoaxEl.textContent = "⚠️ " + data.matched_hoaxes.join(", ");
            hoaxEl.style.color = "#ef4444";
        } else {
            hoaxEl.textContent = "None detected";
            hoaxEl.style.color = "";
        }
    }

    // Feature breakdown
    document.getElementById("feat-xlm-fake").textContent = data.xlm_fake_prob;
    document.getElementById("feat-xlm-real").textContent = data.xlm_real_prob;
    document.getElementById("feat-senti-neg").textContent = data.sentiment_neg_prob;
    document.getElementById("feat-senti-neu").textContent = data.sentiment_neu_prob;
    document.getElementById("feat-senti-pos").textContent = data.sentiment_pos_prob;
    document.getElementById("feat-source-shift").textContent = (data.source_score - 0.50).toFixed(2);
}

// Live News Radar Feed
function setLiveTopic(topic) {
    document.getElementById("live-query").value = topic;
    fetchLiveFeed();
}

async function fetchLiveFeed() {
    const query = document.getElementById("live-query").value.trim() || "India News";
    const container = document.getElementById("live-feed-container");

    container.innerHTML = `<div class="loading-state"><div class="spinner"></div><p style="margin-top:10px;">Fetching latest Indian headlines & running AI radar...</p></div>`;

    try {
        const res = await fetch(`/api/live_feed?query=${encodeURIComponent(query)}`);
        const data = await res.json();

        if (res.ok && data.articles && data.articles.length > 0) {
            container.innerHTML = data.articles.map(art => {
                const a = art.analysis;
                const isFake = a.verdict === "FAKE";
                return `
                    <div class="feed-card">
                        <div class="feed-info">
                            <h3>${escapeHtml(art.title)}</h3>
                            <p>
                                <span>🏛️ ${escapeHtml(art.source || "Unknown")}</span>
                                <span>📅 ${escapeHtml(art.pub_date.substring(0, 16))}</span>
                                <span>🗣️ ${a.language}</span>
                            </p>
                        </div>
                        <div class="feed-badge ${isFake ? 'fake' : 'real'}">
                            ${isFake ? '🔴 High Risk' : '🟢 Verified Likely'}<br>
                            <small>${a.fake_probability} Fake</small>
                        </div>
                    </div>
                `;
            }).join("");
        } else {
            container.innerHTML = `<div class="loading-state">No recent live articles found for this topic.</div>`;
        }
    } catch (e) {
        container.innerHTML = `<div class="loading-state">Could not connect to live feed: ${e.message}</div>`;
    }
}

function escapeHtml(str) {
    return str ? str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;") : "";
}
