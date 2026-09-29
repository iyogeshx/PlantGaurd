/* ═══════════════════════════════════════════════════════════
   PlantGuard AI — Frontend Application Logic
   ═══════════════════════════════════════════════════════════ */

document.addEventListener('DOMContentLoaded', () => {
    // ─── Initialize Core Modules ────────────────────────
    createParticles();
    animateCounters();
    initSampleLeaves();
    initUploadAndCamera();
    initVisualFilters();
    initDiseaseEncyclopedia();
    initRecentScans();
    initReportExport();
    initWeather();
    initChatbot();
    initNavigation();
});

// Global state
let currentFile = null;
let currentBase64 = null;
let currentSampleId = null;
let mediaStream = null;
let currentFacingMode = 'environment';
let allDiseasesData = [];


// ═══════════════════════════════════════════════════════════
// 1. Floating Ambient Particles
// ═══════════════════════════════════════════════════════════
function createParticles() {
    const container = document.getElementById('particles');
    if (!container) return;
    const count = 28;
    for (let i = 0; i < count; i++) {
        const particle = document.createElement('div');
        particle.className = 'particle';
        particle.style.left = Math.random() * 100 + '%';
        particle.style.animationDelay = Math.random() * 16 + 's';
        particle.style.animationDuration = (12 + Math.random() * 8) + 's';
        const size = (2 + Math.random() * 3.5);
        particle.style.width = size + 'px';
        particle.style.height = size + 'px';
        particle.style.opacity = (0.15 + Math.random() * 0.45).toString();
        container.appendChild(particle);
    }
}


// ═══════════════════════════════════════════════════════════
// 2. Animated Stats Counters
// ═══════════════════════════════════════════════════════════
function animateCounters() {
    const counters = document.querySelectorAll('.stat-number');
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const el = entry.target;
                const target = parseInt(el.dataset.target, 10);
                animateNumber(el, 0, target, 1800);
                observer.unobserve(el);
            }
        });
    }, { threshold: 0.4 });

    counters.forEach(counter => observer.observe(counter));
}

function animateNumber(el, start, end, duration) {
    const range = end - start;
    const startTime = performance.now();

    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3);
        const current = Math.floor(start + range * eased);

        if (end >= 1000) {
            el.textContent = current.toLocaleString() + '+';
        } else {
            el.textContent = current;
        }

        if (progress < 1) {
            requestAnimationFrame(update);
        }
    }
    requestAnimationFrame(update);
}


// ═══════════════════════════════════════════════════════════
// 3. Instant Sample Leaves Carousel
// ═══════════════════════════════════════════════════════════
async function initSampleLeaves() {
    const carousel = document.getElementById('samplesCarousel');
    if (!carousel) return;

    try {
        const response = await fetch('/api/samples');
        const samples = await response.json();
        carousel.innerHTML = '';

        samples.forEach((sample) => {
            const chip = document.createElement('div');
            chip.className = 'sample-chip';
            chip.dataset.sampleId = sample.id;
            chip.innerHTML = `
                <img src="${sample.image_url}" alt="${sample.title}" class="sample-thumb" loading="lazy">
                <div class="sample-info">
                    <span class="sample-name">${sample.title}</span>
                    <span class="sample-badge">${sample.badge}</span>
                </div>
            `;

            chip.addEventListener('click', () => {
                selectSampleLeaf(sample, chip);
            });

            carousel.appendChild(chip);
        });
    } catch (e) {
        console.warn('Could not load sample leaves:', e);
    }
}

function selectSampleLeaf(sample, chipEl) {
    // Mark chip as active
    document.querySelectorAll('.sample-chip').forEach(c => c.classList.remove('active'));
    if (chipEl) chipEl.classList.add('active');

    // Update state
    currentFile = null;
    currentBase64 = null;
    currentSampleId = sample.id;

    // Display image in preview
    const previewImage = document.getElementById('previewImage');
    const uploadContent = document.getElementById('uploadContent');
    const previewContainer = document.getElementById('previewContainer');
    const analyzeBtn = document.getElementById('analyzeBtn');

    previewImage.src = sample.image_url;
    uploadContent.style.display = 'none';
    previewContainer.style.display = 'flex';
    analyzeBtn.disabled = false;

    // Scroll to detector
    document.getElementById('detect')?.scrollIntoView({ behavior: 'smooth' });

    // Show toast
    showToast(`Loaded sample: ${sample.title}. Click "Analyze Leaf" or scan!`);
}


// ═══════════════════════════════════════════════════════════
// 4. Upload, Drag & Drop, and Camera Scanner
// ═══════════════════════════════════════════════════════════
function initUploadAndCamera() {
    const uploadZone = document.getElementById('uploadZone');
    const fileInput = document.getElementById('fileInput');
    const browseBtn = document.getElementById('browseBtn');
    const directCameraBtn = document.getElementById('directCameraBtn');
    const tabUpload = document.getElementById('tabUpload');
    const tabCamera = document.getElementById('tabCamera');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const removeBtn = document.getElementById('removeBtn');
    const newAnalysisBtn = document.getElementById('newAnalysisBtn');
    const previewContainer = document.getElementById('previewContainer');
    const previewImage = document.getElementById('previewImage');
    const uploadContent = document.getElementById('uploadContent');
    const resultsCard = document.getElementById('resultsCard');

    // Camera elements
    const cameraModal = document.getElementById('cameraModal');
    const cameraVideo = document.getElementById('cameraVideo');
    const cameraCanvas = document.getElementById('cameraCanvas');
    const shutterBtn = document.getElementById('shutterBtn');
    const switchCameraBtn = document.getElementById('switchCameraBtn');
    const closeCameraBtn = document.getElementById('closeCameraBtn');
    const cancelCameraBtn = document.getElementById('cancelCameraBtn');

    // File Browser Triggers
    browseBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        fileInput.click();
    });

    uploadZone.addEventListener('click', (e) => {
        if (e.target !== removeBtn && !removeBtn.contains(e.target) && !e.target.closest('.visualizer-toolbar')) {
            if (previewContainer.style.display !== 'flex') {
                fileInput.click();
            }
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleSelectedFile(e.target.files[0]);
        }
    });

    // Drag & Drop
    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('drag-over');
    });

    uploadZone.addEventListener('dragleave', () => {
        uploadZone.classList.remove('drag-over');
    });

    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('drag-over');
        if (e.dataTransfer.files.length > 0) {
            handleSelectedFile(e.dataTransfer.files[0]);
        }
    });

    function handleSelectedFile(file) {
        if (!file.type.startsWith('image/')) {
            showToast('Please select a valid image file (JPG, PNG, or WebP)');
            return;
        }

        currentFile = file;
        currentBase64 = null;
        currentSampleId = null;
        document.querySelectorAll('.sample-chip').forEach(c => c.classList.remove('active'));

        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            uploadContent.style.display = 'none';
            previewContainer.style.display = 'flex';
            analyzeBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    // Reset / Remove Image
    removeBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        resetUpload();
    });

    newAnalysisBtn.addEventListener('click', () => {
        resetUpload();
        document.getElementById('detect')?.scrollIntoView({ behavior: 'smooth' });
    });

    function resetUpload() {
        currentFile = null;
        currentBase64 = null;
        currentSampleId = null;
        fileInput.value = '';
        previewImage.src = '';
        previewImage.className = '';
        uploadContent.style.display = 'flex';
        previewContainer.style.display = 'none';
        analyzeBtn.disabled = true;
        resultsCard.style.display = 'none';
        document.querySelectorAll('.sample-chip').forEach(c => c.classList.remove('active'));
    }

    // ─── Camera / Webcam Integration ────────────────────
    async function openCamera() {
        cameraModal.style.display = 'flex';
        try {
            if (mediaStream) {
                mediaStream.getTracks().forEach(track => track.stop());
            }

            const constraints = {
                video: {
                    facingMode: currentFacingMode,
                    width: { ideal: 1280 },
                    height: { ideal: 720 }
                }
            };

            mediaStream = await navigator.mediaDevices.getUserMedia(constraints);
            cameraVideo.srcObject = mediaStream;
        } catch (err) {
            console.error('Camera access error:', err);
            showToast('Camera error: Check browser permissions or device availability.');
            closeCamera();
        }
    }

    function closeCamera() {
        if (mediaStream) {
            mediaStream.getTracks().forEach(track => track.stop());
            mediaStream = null;
        }
        cameraVideo.srcObject = null;
        cameraModal.style.display = 'none';
    }

    // Direct Camera and Tab click
    directCameraBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        openCamera();
    });

    tabCamera.addEventListener('click', () => {
        tabCamera.classList.add('active');
        tabUpload.classList.remove('active');
        openCamera();
    });

    tabUpload.addEventListener('click', () => {
        tabUpload.classList.add('active');
        tabCamera.classList.remove('active');
    });

    closeCameraBtn.addEventListener('click', closeCamera);
    cancelCameraBtn.addEventListener('click', closeCamera);

    switchCameraBtn.addEventListener('click', () => {
        currentFacingMode = (currentFacingMode === 'environment') ? 'user' : 'environment';
        openCamera();
    });

    // Capture Snapshot from Camera
    shutterBtn.addEventListener('click', () => {
        if (!cameraVideo.videoWidth) return;

        cameraCanvas.width = cameraVideo.videoWidth;
        cameraCanvas.height = cameraVideo.videoHeight;
        const ctx = cameraCanvas.getContext('2d');
        ctx.drawImage(cameraVideo, 0, 0, cameraCanvas.width, cameraCanvas.height);

        const b64Data = cameraCanvas.toDataURL('image/jpeg', 0.95);
        currentBase64 = b64Data;
        currentFile = null;
        currentSampleId = null;

        previewImage.src = b64Data;
        uploadContent.style.display = 'none';
        previewContainer.style.display = 'flex';
        analyzeBtn.disabled = false;

        closeCamera();
        showToast('Leaf photo captured! Ready for analysis.');
    });

    // ─── Execute Pathology Analysis ──────────────────────
    analyzeBtn.addEventListener('click', async () => {
        if (!currentFile && !currentBase64 && !currentSampleId) return;

        const btnContent = analyzeBtn.querySelector('.btn-content');
        const btnLoading = analyzeBtn.querySelector('.btn-loading');
        const scanLaser = document.getElementById('scanLaser');
        const scanHud = document.getElementById('scanHud');
        const hudStatusText = document.getElementById('hudStatusText');

        // Start scanning animation & HUD
        btnContent.style.display = 'none';
        btnLoading.style.display = 'flex';
        analyzeBtn.disabled = true;
        scanLaser.style.display = 'block';
        scanHud.style.display = 'block';

        const hudSteps = [
            'Extracting leaf boundary matrix...',
            'Filtering chlorophyll reflectance...',
            'Identifying necrotic lesion centers...',
            'Evaluating MobileNetV2 confidence...'
        ];

        let stepIndex = 0;
        const hudInterval = setInterval(() => {
            stepIndex = (stepIndex + 1) % hudSteps.length;
            if (hudStatusText) hudStatusText.textContent = hudSteps[stepIndex];
        }, 500);

        try {
            let response;
            if (currentFile) {
                const formData = new FormData();
                formData.append('image', currentFile);
                response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });
            } else if (currentBase64) {
                response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ image_base64: currentBase64 })
                });
            } else if (currentSampleId) {
                response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ sample_id: currentSampleId })
                });
            }

            const data = await response.json();
            if (data.error) {
                throw new Error(data.error);
            }

            displayResults(data);
            saveToRecentScans(data);

        } catch (error) {
            console.error('Prediction failed:', error);
            showToast('Analysis Error: ' + error.message);
        } finally {
            clearInterval(hudInterval);
            scanLaser.style.display = 'none';
            scanHud.style.display = 'none';
            btnContent.style.display = 'flex';
            btnLoading.style.display = 'none';
            analyzeBtn.disabled = false;
        }
    });
}


// ═══════════════════════════════════════════════════════════
// 5. Visualizer Filters (Normal, Lesion, Edge)
// ═══════════════════════════════════════════════════════════
function initVisualFilters() {
    const previewImage = document.getElementById('previewImage');
    const filterChips = document.querySelectorAll('.filter-chip');

    filterChips.forEach(chip => {
        chip.addEventListener('click', (e) => {
            e.stopPropagation();
            filterChips.forEach(c => c.classList.remove('active'));
            chip.classList.add('active');

            const filter = chip.dataset.filter;
            previewImage.className = '';
            if (filter === 'lesion') {
                previewImage.classList.add('filter-lesion');
            } else if (filter === 'edge') {
                previewImage.classList.add('filter-edge');
            }
        });
    });
}


// ═══════════════════════════════════════════════════════════
// 6. Display Results & Differential Diagnosis
// ═══════════════════════════════════════════════════════════
function displayResults(data) {
    const resultsCard = document.getElementById('resultsCard');
    const pred = data.prediction;

    // Confidence
    const confidenceVal = (pred.confidence * 100).toFixed(1);
    document.getElementById('confidenceValue').textContent = confidenceVal + '%';

    const confidenceBadge = document.getElementById('confidenceBadge');
    if (pred.confidence >= 0.75) {
        confidenceBadge.style.background = 'rgba(52, 211, 153, 0.12)';
        confidenceBadge.style.borderColor = 'rgba(52, 211, 153, 0.35)';
    } else if (pred.confidence >= 0.45) {
        confidenceBadge.style.background = 'rgba(245, 158, 11, 0.12)';
        confidenceBadge.style.borderColor = 'rgba(245, 158, 11, 0.35)';
    } else {
        confidenceBadge.style.background = 'rgba(239, 68, 68, 0.12)';
        confidenceBadge.style.borderColor = 'rgba(239, 68, 68, 0.35)';
    }

    // Prediction Details
    document.getElementById('predPlant').textContent = pred.plant;
    document.getElementById('predDisease').textContent = pred.disease;
    document.getElementById('predPathogen').textContent = pred.pathogen || 'Foliar Condition';
    document.getElementById('predDescription').textContent = pred.description;

    // ─── Immediate Precaution Advisory & Bio-Security Protocol ───
    const precCard = document.getElementById('precautionCard');
    const precBadge = document.getElementById('precautionBadge');
    const precList = document.getElementById('predPrecautionsList');
    const precBanner = document.getElementById('precautionBanner');
    const precBannerText = document.getElementById('precautionBannerText');
    const precIconBox = document.getElementById('precautionIconBox');

    const precLevel = pred.precaution_level || (pred.severity === 'high' ? 'CRITICAL URGENCY' : (pred.severity === 'none' ? 'ROUTINE CARE' : 'MODERATE CAUTION'));
    precBadge.textContent = precLevel;
    
    // Urgency Card Styling & Colors
    precCard.className = 'precaution-card';
    if (precLevel.includes('CRITICAL')) {
        precCard.classList.add('level-critical');
        precBadge.style.color = '#ef4444';
        precBadge.style.borderColor = 'rgba(239, 68, 68, 0.4)';
        precBadge.style.background = 'rgba(239, 68, 68, 0.15)';
        if (precIconBox) { precIconBox.style.color = '#ef4444'; precIconBox.style.background = 'rgba(239, 68, 68, 0.15)'; }
    } else if (precLevel.includes('HIGH')) {
        precCard.classList.add('level-high');
        precBadge.style.color = '#f97316';
        precBadge.style.borderColor = 'rgba(249, 115, 22, 0.4)';
        precBadge.style.background = 'rgba(249, 115, 22, 0.15)';
        if (precIconBox) { precIconBox.style.color = '#f97316'; precIconBox.style.background = 'rgba(249, 115, 22, 0.15)'; }
    } else if (precLevel.includes('ROUTINE')) {
        precCard.classList.add('level-routine');
        precBadge.style.color = '#10b981';
        precBadge.style.borderColor = 'rgba(16, 185, 129, 0.4)';
        precBadge.style.background = 'rgba(16, 185, 129, 0.15)';
        if (precIconBox) { precIconBox.style.color = '#10b981'; precIconBox.style.background = 'rgba(16, 185, 129, 0.15)'; }
    } else {
        precCard.classList.add('level-moderate');
        precBadge.style.color = '#f59e0b';
        precBadge.style.borderColor = 'rgba(245, 158, 11, 0.4)';
        precBadge.style.background = 'rgba(245, 158, 11, 0.15)';
        if (precIconBox) { precIconBox.style.color = '#f59e0b'; precIconBox.style.background = 'rgba(245, 158, 11, 0.15)'; }
    }

    // Populate Interactive Precaution Checklist
    precList.innerHTML = '';
    const precautions = (pred.precautions && pred.precautions.length > 0) ? pred.precautions : [
        'Isolate symptomatic foliage immediately to stop airborne/splash spore dispersal.',
        'Sanitize all garden shears, knives, and gloves with 70% alcohol between cuts.',
        'Switch strictly to root-level drip irrigation; keep all leaf surfaces completely dry.',
        'Bag and seal infected plant residue in plastic; do NOT compost diseased tissue.'
    ];

    precautions.forEach((p) => {
        const li = document.createElement('li');
        li.className = 'precaution-item';
        li.innerHTML = `
            <div class="precaution-checkbox">✓</div>
            <span class="precaution-text">${p}</span>
        `;
        li.addEventListener('click', () => {
            li.classList.toggle('checked');
            if (li.classList.contains('checked')) {
                showToast('Precaution verified & checked off! ✓');
            }
        });
        precList.appendChild(li);
    });

    // Bio-Security Banner Alert Rule
    if (pred.severity === 'high') {
        precBannerText.textContent = 'Bio-Security Emergency: High-risk airborne pathogen or viral vector. Wear gloves and clean footwear before entering healthy crop rows.';
        precBanner.style.color = '#ef4444';
        precBanner.style.borderColor = 'rgba(239, 68, 68, 0.4)';
        precBanner.style.background = 'rgba(239, 68, 68, 0.08)';
    } else if (pred.severity === 'none') {
        precBannerText.textContent = 'Crop Resilience Tip: Healthy foliage maintains maximum disease defense through root-zone watering and clean mulch barriers.';
        precBanner.style.color = '#10b981';
        precBanner.style.borderColor = 'rgba(16, 185, 129, 0.4)';
        precBanner.style.background = 'rgba(16, 185, 129, 0.08)';
    } else {
        precBannerText.textContent = 'Sanitation Rule: Disinfect shears with 70% alcohol and irrigate only at soil level to prevent water-borne spores.';
        precBanner.style.color = '#f59e0b';
        precBanner.style.borderColor = 'rgba(245, 158, 11, 0.4)';
        precBanner.style.background = 'rgba(245, 158, 11, 0.08)';
    }

    // Symptoms List
    const symptomsBlock = document.getElementById('symptomsBlock');
    const symptomsList = document.getElementById('predSymptoms');
    symptomsList.innerHTML = '';
    if (pred.symptoms && pred.symptoms.length > 0) {
        pred.symptoms.forEach(sym => {
            const li = document.createElement('li');
            li.textContent = sym;
            symptomsList.appendChild(li);
        });
        symptomsBlock.style.display = 'block';
    } else {
        symptomsBlock.style.display = 'none';
    }

    // Environmental Causes
    const causesBlock = document.getElementById('causesBlock');
    if (pred.causes) {
        document.getElementById('predCauses').textContent = pred.causes;
        causesBlock.style.display = 'block';
    } else {
        causesBlock.style.display = 'none';
    }

    // Treatments
    document.getElementById('predOrganicTreatment').textContent = pred.organic_treatment || 'Apply clean compost mulch, avoid water on leaves, and ensure adequate spacing.';
    document.getElementById('predTreatment').textContent = pred.treatment || 'Consult local agricultural extension for registered fungicides/bactericides.';

    // Prevention
    const preventionBlock = document.getElementById('preventionBlock');
    if (pred.prevention) {
        document.getElementById('predPrevention').textContent = pred.prevention;
        preventionBlock.style.display = 'block';
    } else {
        preventionBlock.style.display = 'none';
    }

    // Icon Status
    const predIcon = document.getElementById('predictionIcon');
    predIcon.className = 'prediction-icon';
    if (pred.severity === 'none') {
        predIcon.classList.add('healthy');
        predIcon.innerHTML = `<svg width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>`;
    } else if (pred.severity === 'high') {
        predIcon.classList.add('severe');
        predIcon.innerHTML = `<svg width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>`;
    } else {
        predIcon.classList.add('diseased');
        predIcon.innerHTML = `<svg width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>`;
    }

    // Severity meter
    const severityLabel = document.getElementById('severityLabel');
    const severityFill = document.getElementById('severityFill');
    const severityMap = {
        'none': { width: '12%', color: '#10b981', label: 'Healthy Tissue' },
        'low': { width: '35%', color: '#84cc16', label: 'Low Severity' },
        'moderate': { width: '65%', color: '#f59e0b', label: 'Moderate Risk' },
        'high': { width: '95%', color: '#ef4444', label: 'High Severity Blight' }
    };
    const sev = severityMap[pred.severity] || severityMap['moderate'];
    severityLabel.textContent = sev.label;
    severityLabel.style.color = sev.color;
    severityFill.style.width = '0%';
    severityFill.style.background = sev.color;

    requestAnimationFrame(() => {
        requestAnimationFrame(() => {
            severityFill.style.width = sev.width;
        });
    });

    // Top 3 Differential Diagnosis List
    const top3Container = document.getElementById('top3Container');
    top3Container.innerHTML = '';
    data.top3.forEach((item, index) => {
        const conf = (item.confidence * 100).toFixed(1);
        const div = document.createElement('div');
        div.className = 'top3-item';
        div.style.animationDelay = (index * 0.08) + 's';
        div.innerHTML = `
            <div class="top3-info">
                <span class="top3-disease">${item.disease}</span>
                <span class="top3-plant">${item.plant}</span>
            </div>
            <span class="top3-confidence">${conf}%</span>
        `;
        top3Container.appendChild(div);
    });

    // Display Card
    resultsCard.style.display = 'block';

    // Smooth scroll into view
    setTimeout(() => {
        resultsCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }, 150);
}


// ═══════════════════════════════════════════════════════════
// 7. Recent Session Scans History (LocalStorage)
// ═══════════════════════════════════════════════════════════
const RECENT_SCANS_KEY = 'plantguard_recent_scans';

function initRecentScans() {
    renderRecentScans();
    const clearBtn = document.getElementById('clearHistoryBtn');
    if (clearBtn) {
        clearBtn.addEventListener('click', () => {
            localStorage.removeItem(RECENT_SCANS_KEY);
            renderRecentScans();
            showToast('Scan history cleared.');
        });
    }
}

function saveToRecentScans(data) {
    try {
        const history = JSON.parse(localStorage.getItem(RECENT_SCANS_KEY) || '[]');
        const entry = {
            id: Date.now(),
            time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
            plant: data.prediction.plant,
            disease: data.prediction.disease,
            confidence: (data.prediction.confidence * 100).toFixed(1) + '%',
            image: data.image
        };

        history.unshift(entry);
        if (history.length > 8) history.pop();
        localStorage.setItem(RECENT_SCANS_KEY, JSON.stringify(history));
        renderRecentScans();
    } catch (e) {
        console.warn('LocalStorage save failed:', e);
    }
}

function renderRecentScans() {
    const tray = document.getElementById('recentScansTray');
    const list = document.getElementById('recentScansList');
    if (!tray || !list) return;

    try {
        const history = JSON.parse(localStorage.getItem(RECENT_SCANS_KEY) || '[]');
        if (history.length === 0) {
            tray.style.display = 'none';
            return;
        }

        tray.style.display = 'block';
        list.innerHTML = '';
        history.forEach(item => {
            const card = document.createElement('div');
            card.className = 'recent-card';
            card.innerHTML = `
                <img src="${item.image}" alt="${item.disease}" class="recent-card-thumb">
                <span class="recent-card-plant">${item.plant}</span>
                <div class="recent-card-disease">${item.disease}</div>
                <div class="recent-card-time">${item.time} • ${item.confidence}</div>
            `;
            list.appendChild(card);
        });
    } catch (e) {
        tray.style.display = 'none';
    }
}


// ═══════════════════════════════════════════════════════════
// 8. Disease Encyclopedia & Dossier Modal
// ═══════════════════════════════════════════════════════════
async function initDiseaseEncyclopedia() {
    const grid = document.getElementById('diseaseGrid');
    const searchInput = document.getElementById('diseaseSearchInput');
    const clearSearchBtn = document.getElementById('clearSearchBtn');
    const filterBtns = document.querySelectorAll('.filter-btn');

    // Dossier Modal Elements
    const modal = document.getElementById('diseaseDetailModal');
    const closeBtn = document.getElementById('closeDossierBtn');
    const dismissBtn = document.getElementById('dossierDismissBtn');
    const testBtn = document.getElementById('dossierTestBtn');

    let currentDossierDisease = null;

    closeBtn.addEventListener('click', () => modal.style.display = 'none');
    dismissBtn.addEventListener('click', () => modal.style.display = 'none');
    modal.addEventListener('click', (e) => {
        if (e.target === modal) modal.style.display = 'none';
    });

    testBtn.addEventListener('click', () => {
        modal.style.display = 'none';
        if (currentDossierDisease) {
            // Find a matching sample leaf
            const chips = document.querySelectorAll('.sample-chip');
            let matchedChip = null;
            chips.forEach(c => {
                if (c.textContent.toLowerCase().includes(currentDossierDisease.disease.toLowerCase())) {
                    matchedChip = c;
                }
            });

            if (matchedChip) {
                matchedChip.click();
            } else {
                document.getElementById('detect')?.scrollIntoView({ behavior: 'smooth' });
                showToast(`Ready to test ${currentDossierDisease.plant} - ${currentDossierDisease.disease}. Drop a leaf!`);
            }
        }
    });

    try {
        const response = await fetch('/api/diseases');
        allDiseasesData = await response.json();
        renderDiseaseCards(allDiseasesData);
    } catch (e) {
        console.error('Failed to load diseases:', e);
    }

    function renderDiseaseCards(diseases) {
        grid.innerHTML = '';
        diseases.forEach((d, i) => {
            const card = document.createElement('div');
            card.className = 'disease-card';
            card.dataset.plant = d.plant.split(' ')[0];
            card.dataset.disease = d.disease;
            card.style.animationDelay = (i * 0.04) + 's';

            const severityClass = `severity-${d.severity}`;
            const severityText = d.severity === 'none' ? 'Healthy' : d.severity.charAt(0).toUpperCase() + d.severity.slice(1);

            card.innerHTML = `
                <div class="disease-card-header">
                    <span class="disease-card-plant">${d.plant}</span>
                    <span class="disease-card-severity ${severityClass}">${severityText}</span>
                </div>
                <div class="disease-card-name">${d.disease}</div>
                <p class="disease-card-desc">${d.description || 'Foliar crop disease.'}</p>
                <div class="disease-card-footer">
                    <span>View Clinical Dossier</span>
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                        <line x1="5" y1="12" x2="19" y2="12"></line>
                        <polyline points="12 5 19 12 12 19"></polyline>
                    </svg>
                </div>
            `;

            card.addEventListener('click', () => {
                openDiseaseDossier(d);
            });

            grid.appendChild(card);
        });
    }

    function openDiseaseDossier(d) {
        currentDossierDisease = d;
        document.getElementById('dossierPlant').textContent = d.plant;
        document.getElementById('dossierTitle').textContent = d.disease;
        document.getElementById('dossierPathogen').textContent = d.pathogen || 'Pathological Condition';
        document.getElementById('dossierDescription').textContent = d.description;

        const sevEl = document.getElementById('dossierSeverity');
        sevEl.textContent = d.severity === 'none' ? 'Healthy Plant' : d.severity.toUpperCase() + ' RISK';
        sevEl.className = `dossier-severity severity-${d.severity}`;

        const symptomsList = document.getElementById('dossierSymptoms');
        symptomsList.innerHTML = '';
        if (d.symptoms && d.symptoms.length > 0) {
            d.symptoms.forEach(s => {
                const li = document.createElement('li');
                li.textContent = s;
                symptomsList.appendChild(li);
            });
        } else {
            symptomsList.innerHTML = '<li>Inspect visual lesions, spotting, or leaf margins.</li>';
        }

        document.getElementById('dossierCauses').textContent = d.causes || 'High relative humidity, splashing rain, or insect vector activity.';
        document.getElementById('dossierOrganic').textContent = d.organic_treatment || 'Apply biological fungicides, neem extracts, and sanitize infected plant matter.';
        document.getElementById('dossierChemical').textContent = d.treatment || 'Consult local agricultural extension for approved chemical controls.';
        document.getElementById('dossierPrevention').textContent = d.prevention || 'Implement crop rotation, drip irrigation, and disease-free seeds.';

        // Populate Dossier Precaution Advice
        const dossierPrecBadge = document.getElementById('dossierPrecautionBadge');
        const dossierPrecList = document.getElementById('dossierPrecautionsList');
        if (dossierPrecBadge && dossierPrecList) {
            const pLevel = d.precaution_level || (d.severity === 'high' ? 'CRITICAL URGENCY' : (d.severity === 'none' ? 'ROUTINE CARE' : 'MODERATE CAUTION'));
            dossierPrecBadge.textContent = pLevel;
            dossierPrecBadge.style.color = d.precaution_color || (d.severity === 'high' ? '#ef4444' : (d.severity === 'none' ? '#10b981' : '#f59e0b'));
            dossierPrecList.innerHTML = '';
            const precs = (d.precautions && d.precautions.length > 0) ? d.precautions : [
                'Sanitize pruning shears with 70% alcohol between cuts.',
                'Avoid overhead irrigation; water exclusively at soil level.',
                'Isolate or remove diseased foliage to prevent airborne or vector transmission.'
            ];
            precs.forEach(p => {
                const li = document.createElement('li');
                li.textContent = p;
                dossierPrecList.appendChild(li);
            });
        }

        modal.style.display = 'flex';
    }

    // Plant Filter Tabs
    filterBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            filterBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            applyFilters();
        });
    });

    // Real-Time Search Input
    searchInput.addEventListener('input', () => {
        if (searchInput.value.trim().length > 0) {
            clearSearchBtn.style.display = 'block';
        } else {
            clearSearchBtn.style.display = 'none';
        }
        applyFilters();
    });

    clearSearchBtn.addEventListener('click', () => {
        searchInput.value = '';
        clearSearchBtn.style.display = 'none';
        applyFilters();
    });

    function applyFilters() {
        const activeFilter = document.querySelector('.filter-btn.active')?.dataset.filter || 'all';
        const query = searchInput.value.toLowerCase().trim();

        document.querySelectorAll('.disease-card').forEach(card => {
            const plant = card.dataset.plant;
            const text = card.textContent.toLowerCase();

            const matchesPlant = (activeFilter === 'all' || plant === activeFilter);
            const matchesQuery = (query === '' || text.includes(query));

            if (matchesPlant && matchesQuery) {
                card.classList.remove('hidden');
            } else {
                card.classList.add('hidden');
            }
        });
    }
}


// ═══════════════════════════════════════════════════════════
// 9. Export Diagnostic Certificate & Share
// ═══════════════════════════════════════════════════════════
function initReportExport() {
    const printBtn = document.getElementById('printReportBtn');
    const shareBtn = document.getElementById('shareResultBtn');

    if (printBtn) {
        printBtn.addEventListener('click', () => {
            window.print();
        });
    }

    if (shareBtn) {
        shareBtn.addEventListener('click', async () => {
            const disease = document.getElementById('predDisease').textContent;
            const plant = document.getElementById('predPlant').textContent;
            const confidence = document.getElementById('confidenceValue').textContent;

            const shareData = {
                title: `PlantGuard AI Diagnostic: ${plant} ${disease}`,
                text: `Diagnosis: ${plant} - ${disease} (${confidence} confidence). Analyzed with PlantGuard AI.`,
                url: window.location.href
            };

            if (navigator.share) {
                try {
                    await navigator.share(shareData);
                } catch (e) {
                    // user cancelled share
                }
            } else {
                // Clipboard copy
                navigator.clipboard.writeText(`${shareData.title}\n${shareData.text}`);
                showToast('Diagnosis report copied to clipboard!');
            }
        });
    }
}


// ═══════════════════════════════════════════════════════════
// 10. Toast Notification Helper
// ═══════════════════════════════════════════════════════════
let toastTimeout = null;
function showToast(msg) {
    const toast = document.getElementById('toast');
    if (!toast) return;
    toast.textContent = msg;
    toast.style.display = 'flex';

    if (toastTimeout) clearTimeout(toastTimeout);
    toastTimeout = setTimeout(() => {
        toast.style.display = 'none';
    }, 3800);
}


// ═══════════════════════════════════════════════════════════
// 11. Navigation Scroll Spy & Smooth Scrolling
// ═══════════════════════════════════════════════════════════
function initNavigation() {
    const sections = ['hero', 'samples', 'weather', 'detect', 'chatbot', 'diseases', 'about'];
    const navLinks = {
        'hero': document.getElementById('nav-detect'),
        'samples': document.getElementById('nav-samples'),
        'weather': document.getElementById('nav-weather'),
        'detect': document.getElementById('nav-detect'),
        'chatbot': document.getElementById('nav-chatbot'),
        'diseases': document.getElementById('nav-diseases'),
        'about': document.getElementById('nav-about')
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
                const link = navLinks[entry.target.id];
                if (link) link.classList.add('active');
            }
        });
    }, { rootMargin: '-35% 0px', threshold: 0 });

    sections.forEach(id => {
        const el = document.getElementById(id);
        if (el) observer.observe(el);
    });

    document.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const target = link.getAttribute('href').substring(1);
            document.getElementById(target)?.scrollIntoView({ behavior: 'smooth' });
        });
    });
}


// ═══════════════════════════════════════════════════════════
// 12. Real-Time Weather & Disease Risk Tracker
// ═══════════════════════════════════════════════════════════
function initWeather() {
    const loadingEl = document.getElementById('weatherLoading');
    const dataEl = document.getElementById('weatherData');
    const retryBtn = document.getElementById('weatherRetryBtn');

    if (!loadingEl || !dataEl) return;

    function fetchWeatherData(lat, lon, locationName = 'Your Local Conditions') {
        loadingEl.style.display = 'flex';
        dataEl.style.display = 'none';

        fetch(`/api/weather?lat=${lat}&lon=${lon}`)
            .then(res => {
                if (!res.ok) throw new Error('Weather data unavailable');
                return res.json();
            })
            .then(data => {
                if (!data.success) throw new Error(data.error || 'Weather analysis failed');

                // 1. Current Conditions
                const tempEl = document.getElementById('weatherTemp');
                const humEl = document.getElementById('weatherHumidity');
                const windEl = document.getElementById('weatherWind');
                const condEl = document.getElementById('weatherCondition');
                const iconBox = document.getElementById('weatherIconBox');
                const locLabel = document.querySelector('.weather-location-label');

                if (tempEl) tempEl.textContent = `${Math.round(data.current.temperature)}°C`;
                if (humEl) humEl.textContent = `${Math.round(data.current.humidity)}%`;
                if (windEl) windEl.textContent = `${Math.round(data.current.wind_speed)} km/h`;
                if (condEl) condEl.textContent = data.current.condition;
                if (locLabel) locLabel.textContent = locationName;

                if (iconBox) {
                    iconBox.innerHTML = getWeatherIconSvg(data.current.weather_code);
                }

                // 2. Disease Risk Gauge
                const risk = data.disease_risk;
                const riskBadge = document.getElementById('riskBadge');
                const riskGaugeFill = document.getElementById('riskGaugeFill');
                const riskAdvice = document.getElementById('riskAdvice');
                const riskFactorsContainer = document.getElementById('riskFactors');

                if (riskBadge) {
                    riskBadge.textContent = `${risk.level} RISK`;
                    riskBadge.style.color = risk.color;
                    riskBadge.style.background = `${risk.color}22`;
                    riskBadge.style.borderColor = `${risk.color}55`;
                }

                if (riskGaugeFill) {
                    riskGaugeFill.style.width = `${Math.max(5, risk.score)}%`;
                    riskGaugeFill.style.background = risk.color;
                }

                if (riskAdvice) {
                    riskAdvice.textContent = risk.advice;
                }

                if (riskFactorsContainer) {
                    riskFactorsContainer.innerHTML = '';
                    if (risk.factors && risk.factors.length > 0) {
                        risk.factors.forEach(factor => {
                            const item = document.createElement('div');
                            item.className = 'risk-factor-item';
                            item.style.borderLeftColor = risk.color;
                            item.innerHTML = `
                                <span class="factor-icon">⚠️</span>
                                <span>${factor}</span>
                            `;
                            riskFactorsContainer.appendChild(item);
                        });
                    }
                }

                // 3. Forecast Strip
                const forecastStrip = document.getElementById('forecastStrip');
                if (forecastStrip && data.forecast) {
                    forecastStrip.innerHTML = '';
                    data.forecast.forEach((day, index) => {
                        const dayCard = document.createElement('div');
                        dayCard.className = 'forecast-day';
                        
                        let dayLabel = 'Day ' + (index + 1);
                        if (day.date) {
                            const dateObj = new Date(day.date + 'T00:00:00');
                            if (!isNaN(dateObj)) {
                                dayLabel = dateObj.toLocaleDateString(undefined, { weekday: 'short', month: 'short', day: 'numeric' });
                            }
                        }

                        dayCard.innerHTML = `
                            <span class="forecast-date">${dayLabel}</span>
                            <span class="forecast-temps">${Math.round(day.temp_max)}° / ${Math.round(day.temp_min)}°C</span>
                            <span class="forecast-rain">💧 ${day.rain_probability}% rain</span>
                        `;
                        forecastStrip.appendChild(dayCard);
                    });
                }

                loadingEl.style.display = 'none';
                dataEl.style.display = 'grid';
            })
            .catch(err => {
                console.warn('Weather fetch fallback:', err);
                if (lat !== 28.6139) {
                    fetchWeatherData(28.6139, 77.2090, 'Regional Farm Baseline');
                } else {
                    loadingEl.style.display = 'flex';
                    const p = loadingEl.querySelector('p');
                    if (p) p.textContent = 'Weather service temporarily unavailable. Click Retry to reconnect.';
                }
            });
    }

    function locateAndFetch() {
        if ('geolocation' in navigator) {
            navigator.geolocation.getCurrentPosition(
                (pos) => {
                    fetchWeatherData(pos.coords.latitude, pos.coords.longitude, 'Your Local Farm Weather');
                },
                (err) => {
                    console.info('Geolocation access not granted, using regional agricultural benchmark.', err);
                    fetchWeatherData(28.6139, 77.2090, 'Regional Farm Baseline (Default)');
                },
                { timeout: 7000 }
            );
        } else {
            fetchWeatherData(28.6139, 77.2090, 'Regional Farm Baseline (Default)');
        }
    }

    if (retryBtn) {
        retryBtn.addEventListener('click', () => {
            locateAndFetch();
        });
    }

    locateAndFetch();
}

function getWeatherIconSvg(code) {
    if (code === 0 || code === 1) {
        return `<svg width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
            <circle cx="12" cy="12" r="4"/>
            <path d="M12 2v2"/><path d="M12 20v2"/><path d="m4.93 4.93 1.41 1.41"/>
            <path d="m17.66 17.66 1.41 1.41"/><path d="M2 12h2"/><path d="M20 12h2"/>
            <path d="m6.34 17.66-1.41 1.41"/><path d="m19.07 4.93-1.41 1.41"/>
        </svg>`;
    } else if (code >= 2 && code <= 3) {
        return `<svg width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
            <path d="M17.5 19H9a7 7 0 1 1 6.71-9h1.79a4.5 4.5 0 1 1 0 9Z"/>
        </svg>`;
    } else if (code >= 51 && code <= 82) {
        return `<svg width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
            <path d="M20 16.58A5 5 0 0 0 18 7h-1.26A8 8 0 1 0 4 15.25"/>
            <path d="m8 19-2 3"/><path d="m12 19-2 3"/><path d="m16 19-2 3"/>
        </svg>`;
    } else if (code >= 95) {
        return `<svg width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
            <path d="M19 16.9A5 5 0 0 0 18 7h-1.26a8 8 0 1 0-11.62 9"/>
            <polyline points="13 11 9 17 15 17 11 23"/>
        </svg>`;
    } else {
        return `<svg width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
            <path d="M17.5 19H9a7 7 0 1 1 6.71-9h1.79a4.5 4.5 0 1 1 0 9Z"/>
        </svg>`;
    }
}


// ═══════════════════════════════════════════════════════════
// ═══════════════════════════════════════════════════════════
// 13. AI Plant Pathologist Chatbot (Embedded Section + Floating Widget)
// ═══════════════════════════════════════════════════════════
function initChatbot() {
    let chatHistory = [];

    // Helper: Escape HTML
    function escapeChatHtml(str) {
        const p = document.createElement('p');
        p.textContent = str;
        return p.innerHTML;
    }

    // Helper: Markdown to HTML formatting
    function formatChatMessage(text) {
        let esc = escapeChatHtml(text);
        esc = esc.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        esc = esc.replace(/\*(.*?)\*/g, '<em>$1</em>');
        esc = esc.replace(/^[•\-]\s+(.*)$/gm, '<li>$1</li>');
        esc = esc.replace(/(<li>.*<\/li>)/gs, '<ul>$1</ul>');
        return esc.split('\n\n').map(chunk => {
            if (chunk.includes('<ul>')) return chunk;
            return `<p>${chunk.replace(/\n/g, '<br>')}</p>`;
        }).join('');
    }

    // ─── A. Dedicated Section Chatbot ───────────────────────
    const sectionMsgs = document.getElementById('sectionChatMessages');
    const sectionInput = document.getElementById('sectionChatInput');
    const sectionSendBtn = document.getElementById('sectionChatSendBtn');
    const sectionClearBtn = document.getElementById('sectionClearChatBtn');
    const sectionPromptBtns = document.querySelectorAll('.section-quick-prompt');

    let sectionSending = false;

    function sendSectionMessage(text) {
        const trimmed = (text || '').trim();
        if (!trimmed || sectionSending || !sectionMsgs) return;

        sectionSending = true;
        if (sectionInput) sectionInput.value = '';
        if (sectionSendBtn) sectionSendBtn.disabled = true;

        // User bubble
        const userDiv = document.createElement('div');
        userDiv.className = 'chat-message user';
        userDiv.innerHTML = `<div class="chat-bubble"><p>${escapeChatHtml(trimmed)}</p></div>`;
        sectionMsgs.appendChild(userDiv);
        sectionMsgs.scrollTop = sectionMsgs.scrollHeight;

        // Typing indicator
        const typingDiv = document.createElement('div');
        typingDiv.className = 'chat-message bot';
        typingDiv.id = 'sectionChatTyping';
        typingDiv.innerHTML = `
            <div class="chat-bubble chat-typing">
                <span class="typing-dot"></span>
                <span class="typing-dot"></span>
                <span class="typing-dot"></span>
            </div>
        `;
        sectionMsgs.appendChild(typingDiv);
        sectionMsgs.scrollTop = sectionMsgs.scrollHeight;

        fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: trimmed, history: chatHistory })
        })
        .then(res => res.json())
        .then(data => {
            typingDiv.remove();
            const reply = data.reply || data.error || 'No response received. Please try again.';
            chatHistory.push({ role: 'user', text: trimmed });
            chatHistory.push({ role: 'model', text: reply });

            const botDiv = document.createElement('div');
            botDiv.className = 'chat-message bot';
            botDiv.innerHTML = `<div class="chat-bubble">${formatChatMessage(reply)}</div>`;
            sectionMsgs.appendChild(botDiv);
            sectionMsgs.scrollTop = sectionMsgs.scrollHeight;
        })
        .catch(err => {
            typingDiv.remove();
            const errDiv = document.createElement('div');
            errDiv.className = 'chat-message bot';
            errDiv.innerHTML = `<div class="chat-bubble"><p>⚠️ Connection error. Please verify your network and try again.</p></div>`;
            sectionMsgs.appendChild(errDiv);
            sectionMsgs.scrollTop = sectionMsgs.scrollHeight;
        })
        .finally(() => {
            sectionSending = false;
            if (sectionSendBtn) sectionSendBtn.disabled = false;
            if (sectionInput) sectionInput.focus();
        });
    }

    if (sectionSendBtn && sectionInput) {
        sectionSendBtn.addEventListener('click', () => sendSectionMessage(sectionInput.value));
        sectionInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendSectionMessage(sectionInput.value);
            }
        });
    }

    if (sectionClearBtn && sectionMsgs) {
        sectionClearBtn.addEventListener('click', () => {
            sectionMsgs.innerHTML = `
                <div class="chat-message bot">
                    <div class="chat-bubble">
                        <p><strong>Chat conversation reset.</strong> How can I assist you with your foliage health today? 🌿</p>
                    </div>
                </div>
            `;
            chatHistory = [];
        });
    }

    sectionPromptBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const prompt = btn.dataset.prompt;
            if (prompt) {
                document.getElementById('chatbot')?.scrollIntoView({ behavior: 'smooth' });
                sendSectionMessage(prompt);
            }
        });
    });


    // ─── B. Floating Assistant Widget ───────────────────────
    const fab = document.getElementById('chatbotFab');
    const floatWindow = document.getElementById('chatbotWindow');
    const floatCloseBtn = document.getElementById('chatbotCloseBtn');
    const floatMsgs = document.getElementById('chatbotMessages');
    const floatInput = document.getElementById('chatInput');
    const floatSendBtn = document.getElementById('chatSendBtn');
    const floatPromptBtns = document.querySelectorAll('.quick-prompt-btn');

    let floatSending = false;

    if (fab && floatWindow) {
        fab.addEventListener('click', () => {
            const isHidden = floatWindow.style.display === 'none' || floatWindow.style.display === '';
            floatWindow.style.display = isHidden ? 'flex' : 'none';
            if (isHidden && floatInput) {
                setTimeout(() => floatInput.focus(), 200);
            }
        });
    }

    if (floatCloseBtn && floatWindow) {
        floatCloseBtn.addEventListener('click', () => {
            floatWindow.style.display = 'none';
        });
    }

    function sendFloatMessage(text) {
        const trimmed = (text || '').trim();
        if (!trimmed || floatSending || !floatMsgs) return;

        floatSending = true;
        if (floatInput) floatInput.value = '';
        if (floatSendBtn) floatSendBtn.disabled = true;

        const userDiv = document.createElement('div');
        userDiv.className = 'chat-message user';
        userDiv.innerHTML = `<div class="chat-bubble"><p>${escapeChatHtml(trimmed)}</p></div>`;
        floatMsgs.appendChild(userDiv);
        floatMsgs.scrollTop = floatMsgs.scrollHeight;

        const typingDiv = document.createElement('div');
        typingDiv.className = 'chat-message bot';
        typingDiv.id = 'floatChatTyping';
        typingDiv.innerHTML = `
            <div class="chat-bubble chat-typing">
                <span class="typing-dot"></span>
                <span class="typing-dot"></span>
                <span class="typing-dot"></span>
            </div>
        `;
        floatMsgs.appendChild(typingDiv);
        floatMsgs.scrollTop = floatMsgs.scrollHeight;

        fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: trimmed, history: chatHistory })
        })
        .then(res => res.json())
        .then(data => {
            typingDiv.remove();
            const reply = data.reply || data.error || 'No response received. Please try again.';
            chatHistory.push({ role: 'user', text: trimmed });
            chatHistory.push({ role: 'model', text: reply });

            const botDiv = document.createElement('div');
            botDiv.className = 'chat-message bot';
            botDiv.innerHTML = `<div class="chat-bubble">${formatChatMessage(reply)}</div>`;
            floatMsgs.appendChild(botDiv);
            floatMsgs.scrollTop = floatMsgs.scrollHeight;
        })
        .catch(err => {
            typingDiv.remove();
            const errDiv = document.createElement('div');
            errDiv.className = 'chat-message bot';
            errDiv.innerHTML = `<div class="chat-bubble"><p>⚠️ Connection error. Please check your network.</p></div>`;
            floatMsgs.appendChild(errDiv);
            floatMsgs.scrollTop = floatMsgs.scrollHeight;
        })
        .finally(() => {
            floatSending = false;
            if (floatSendBtn) floatSendBtn.disabled = false;
            if (floatInput) floatInput.focus();
        });
    }

    if (floatSendBtn && floatInput) {
        floatSendBtn.addEventListener('click', () => sendFloatMessage(floatInput.value));
        floatInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendFloatMessage(floatInput.value);
            }
        });
    }

    floatPromptBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const prompt = btn.dataset.prompt;
            if (prompt) sendFloatMessage(prompt);
        });
    });


    // ─── C. Global Triggers across UI ───────────────────────
    // 1. Navbar "AI Chatbot" button
    const navChatCta = document.getElementById('navChatCta');
    if (navChatCta) {
        navChatCta.addEventListener('click', (e) => {
            e.preventDefault();
            document.getElementById('chatbot')?.scrollIntoView({ behavior: 'smooth' });
            setTimeout(() => {
                document.getElementById('sectionChatInput')?.focus();
            }, 500);
        });
    }

    // 2. Diagnosis Result Card "Consult AI on Diagnosis" button
    const askAiResultBtn = document.getElementById('askAiResultBtn');
    if (askAiResultBtn) {
        askAiResultBtn.addEventListener('click', () => {
            const plant = document.getElementById('predPlant')?.textContent || 'crop';
            const disease = document.getElementById('predDisease')?.textContent || 'detected foliar condition';
            const confidence = document.getElementById('confidenceValue')?.textContent || '';
            const prompt = `My ${plant} plant was diagnosed with ${disease} (${confidence} confidence). What are the critical immediate treatment steps, organic remedies, and prevention protocols I should follow?`;

            const chatSection = document.getElementById('chatbot');
            if (chatSection) {
                chatSection.scrollIntoView({ behavior: 'smooth' });
                setTimeout(() => {
                    sendSectionMessage(prompt);
                }, 500);
            } else if (floatWindow) {
                floatWindow.style.display = 'flex';
                sendFloatMessage(prompt);
            }
        });
    }

    // 3. Clinical Dossier Modal "Ask AI About This Condition" button
    const dossierChatBtn = document.getElementById('dossierChatBtn');
    if (dossierChatBtn) {
        dossierChatBtn.addEventListener('click', () => {
            const plant = document.getElementById('dossierPlant')?.textContent || 'crop';
            const disease = document.getElementById('dossierTitle')?.textContent || 'disease';
            const prompt = `Can you provide a comprehensive agronomic guide for ${plant} - ${disease}, including symptom progression, biological fungicides, and vector prevention?`;

            // Close modal
            const modal = document.getElementById('diseaseDetailModal');
            if (modal) modal.style.display = 'none';

            const chatSection = document.getElementById('chatbot');
            if (chatSection) {
                chatSection.scrollIntoView({ behavior: 'smooth' });
                setTimeout(() => {
                    sendSectionMessage(prompt);
                }, 500);
            } else if (floatWindow) {
                floatWindow.style.display = 'flex';
                sendFloatMessage(prompt);
            }
        });
    }
}
