const form = document.getElementById('transcriptForm');
const videoUrl = document.getElementById('videoUrl');
const langSelect = document.getElementById('langSelect');
const timestampsChk = document.getElementById('timestampsChk');
const fetchBtn = document.getElementById('fetchBtn');
const fetchBtnText = document.getElementById('fetchBtnText');
const loadingSpinner = document.getElementById('loadingSpinner');
const errorBox = document.getElementById('errorBox');

const transcriptBox = document.getElementById('transcriptBox');
const summaryBox = document.getElementById('summaryBox');
const copyTranscriptBtn = document.getElementById('copyTranscriptBtn');
const downloadTranscriptBtn = document.getElementById('downloadTranscriptBtn');
const summarizeBtn = document.getElementById('summarizeBtn');
const copySummaryBtn = document.getElementById('copySummaryBtn');
const downloadSummaryBtn = document.getElementById('downloadSummaryBtn');
const maxSentencesInput = document.getElementById('maxSentences');
const clearBtn = document.getElementById('clearBtn');

const themeToggle = document.getElementById('themeToggle');
const yearEl = document.getElementById('year');
yearEl.textContent = new Date().getFullYear();

// History-related elements
const historyToggle = document.getElementById('historyToggle');
const historySidebar = document.getElementById('historySidebar');
const historyOverlay = document.getElementById('historyOverlay');
const closeHistory = document.getElementById('closeHistory');
const historyList = document.getElementById('historyList');
const clearHistoryBtn = document.getElementById('clearHistory');

// Video meta elements
const videoMeta = document.getElementById('videoMeta');
const videoThumb = document.getElementById('videoThumb');
const videoTitle = document.getElementById('videoTitle');
const videoChannel = document.getElementById('videoChannel');
const openLinkBtn = document.getElementById('openLinkBtn');

// History management
let transcriptHistory = JSON.parse(localStorage.getItem('transcriptHistory') || '[]');

function setLoading(state) {
  if (state) {
    fetchBtn.disabled = true;
    fetchBtn.classList.add('opacity-70','cursor-not-allowed');
    loadingSpinner.classList.remove('hidden');
    fetchBtnText.textContent = 'Loading...';
  } else {
    fetchBtn.disabled = false;
    fetchBtn.classList.remove('opacity-70','cursor-not-allowed');
    loadingSpinner.classList.add('hidden');
    fetchBtnText.textContent = 'Get Transcript';
  }
}

function showError(msg) {
  errorBox.textContent = msg;
  errorBox.classList.remove('hidden');
}

function clearError() {
  errorBox.classList.add('hidden');
  errorBox.textContent = '';
}

async function fetchTranscript(e) {
  e.preventDefault();
  clearError();
  summaryBox.value = '';
  const url = videoUrl.value.trim();
  if (!url) {
    showError('Provide a YouTube link.');
    return;
  }
  setLoading(true);
  try {
    const res = await fetch('/get_transcript', {
      method: 'POST',
      headers: { 'Content-Type':'application/json' },
      body: JSON.stringify({
        url,
        lang: langSelect.value.trim(),
        timestamps: timestampsChk.checked
      })
    });
    const data = await res.json();
    if (!res.ok) {
      showError(data.error || 'Failed.');
      return;
    }
    transcriptBox.value = data.transcript || '';

    // set video metadata UI
    if (data.thumbnail) {
      videoThumb.src = data.thumbnail;
      videoThumb.classList.remove('hidden');
    } else {
      videoThumb.src = '';
      videoThumb.classList.add('hidden');
    }
    videoTitle.textContent = data.title || '';
    videoChannel.textContent = data.channel || '';

    // set open link button
    if (url) {
      openLinkBtn.href = url;
      openLinkBtn.classList.remove('hidden');
    } else {
      openLinkBtn.href = '#';
      openLinkBtn.classList.add('hidden');
    }

    videoMeta.classList.remove('hidden');

    // Save to history (include thumbnail & channel)
    saveToHistory({
      id: Date.now(),
      url: url,
      title: data.title || 'YouTube Video',
      thumbnail: data.thumbnail || '',
      channel: data.channel || '',
      transcript: data.transcript || '',
      summary: '',
      language: langSelect.value.trim(),
      timestamp: new Date().toISOString(),
      hasTimestamps: timestampsChk.checked
    });

  } catch (err) {
    showError('Network error.');
  } finally {
    setLoading(false);
  }
}

async function summarize() {
  clearError();
  const transcript = transcriptBox.value.trim();
  if (!transcript) {
    showError('No transcript to summarize.');
    return;
  }
  summarizeBtn.disabled = true;
  summarizeBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i><span class="hidden md:inline">Summarizing...</span>';
  try {
    const res = await fetch('/summarize_llm', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({
        transcript,
        max_sentences: parseInt(maxSentencesInput.value || '8', 10)
      })
    });
    const data = await res.json();
    if (!res.ok) {
      showError(data.error || 'LLM summarization failed.');
      return;
    }
    summaryBox.value = data.summary || '';
    
    // Update history with summary
    updateHistoryWithSummary(transcript, data.summary || '');
    
  } catch (e) {
    showError('Summarization error.');
  } finally {
    summarizeBtn.disabled = false;
    summarizeBtn.innerHTML = '<i class="fa-solid fa-bolt"></i><span class="hidden md:inline">Summarize</span>';
  }
}

// History functions
function saveToHistory(item) {
  // Remove duplicates based on URL
  transcriptHistory = transcriptHistory.filter(h => h.url !== item.url);
  
  // Add to beginning
  transcriptHistory.unshift(item);
  
  // Keep only last 50 items
  transcriptHistory = transcriptHistory.slice(0, 50);
  
  // Save to localStorage
  localStorage.setItem('transcriptHistory', JSON.stringify(transcriptHistory));
  
  // Update UI if sidebar is open
  if (!historySidebar.classList.contains('translate-x-full')) {
    renderHistory();
  }
}

function updateHistoryWithSummary(transcript, summary) {
  const item = transcriptHistory.find(h => h.transcript === transcript);
  if (item) {
    item.summary = summary;
    localStorage.setItem('transcriptHistory', JSON.stringify(transcriptHistory));
    // Update UI if sidebar is open
    if (!historySidebar.classList.contains('translate-x-full')) {
      renderHistory();
    }
  }
}

function renderHistory() {
  if (transcriptHistory.length === 0) {
    historyList.innerHTML = `
      <div class="text-center text-text/60 py-8">
        <i class="fa-solid fa-folder-open text-3xl mb-2 opacity-50"></i>
        <p>No transcripts yet</p>
      </div>
    `;
    return;
  }
  
  historyList.innerHTML = transcriptHistory.map(item => {
    const date = new Date(item.timestamp);
    const timeAgo = formatTimeAgo(date);
    const preview = item.transcript.slice(0, 150) + (item.transcript.length > 150 ? '...' : '');
    const thumb = item.thumbnail ? `<img src="${escapeHtml(item.thumbnail)}" class="history-thumb rounded" alt="thumb">` : `<div class="history-thumb bg-layer2 rounded"></div>`;
    
    return `
      <div class="history-item" data-id="${item.id}">
        <div class="flex gap-3 items-start">
          <div class="flex-shrink-0">${thumb}</div>
          <div class="min-w-0">
            <div class="history-item-title">${escapeHtml(item.title)}</div>
            <div class="history-item-meta">
              <span><i class="fa-regular fa-clock"></i> ${timeAgo}</span>
              <span><i class="fa-solid fa-globe"></i> ${item.language.toUpperCase()}</span>
              ${item.summary ? '<span><i class="fa-solid fa-check text-accent"></i> Summarized</span>' : ''}
            </div>
            <div class="history-item-preview">${escapeHtml(preview)}</div>
          </div>
        </div>
      </div>
    `;
  }).join('');
  
  // Add click listeners
  document.querySelectorAll('.history-item').forEach(item => {
    item.addEventListener('click', () => {
      const id = parseInt(item.dataset.id);
      loadHistoryItem(id);
    });
  });
}

function loadHistoryItem(id) {
  const item = transcriptHistory.find(h => h.id === id);
  if (!item) return;

  // Load data into form
  videoUrl.value = item.url;
  langSelect.value = item.language;
  timestampsChk.checked = item.hasTimestamps;
  transcriptBox.value = item.transcript;
  summaryBox.value = item.summary || '';

  // load meta
  if (item.thumbnail) {
    videoThumb.src = item.thumbnail;
    videoThumb.classList.remove('hidden');
  } else {
    videoThumb.src = '';
    videoThumb.classList.add('hidden');
  }
  videoTitle.textContent = item.title || '';
  videoChannel.textContent = item.channel || '';

  // set open link button to this history item's URL
  if (item.url) {
    openLinkBtn.href = item.url;
    openLinkBtn.classList.remove('hidden');
  } else {
    openLinkBtn.href = '#';
    openLinkBtn.classList.add('hidden');
  }
  videoMeta.classList.remove('hidden');

  // Close sidebar
  closeHistorySidebar();

  // Smooth scroll to top
  window.scrollTo({ top: 0, behavior: 'smooth' });

  // Show success message
  clearError();
  showTemporaryMessage('History item loaded successfully!');
}

function formatTimeAgo(date) {
  const now = new Date();
  const diffMs = now - date;
  const diffMins = Math.floor(diffMs / 60000);
  const diffHours = Math.floor(diffMins / 60);
  const diffDays = Math.floor(diffHours / 24);
  
  if (diffMins < 1) return 'Just now';
  if (diffMins < 60) return `${diffMins}m ago`;
  if (diffHours < 24) return `${diffHours}h ago`;
  if (diffDays < 7) return `${diffDays}d ago`;
  return date.toLocaleDateString();
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

function clearHistory() {
  if (confirm('Are you sure you want to clear all transcript history?')) {
    transcriptHistory = [];
    localStorage.removeItem('transcriptHistory');
    renderHistory();
    showTemporaryMessage('History cleared!');
  }
}

function openHistorySidebar() {
  historySidebar.classList.remove('translate-x-full');
  historyOverlay.classList.remove('hidden');
  document.body.style.overflow = 'hidden';
  renderHistory();
}

function closeHistorySidebar() {
  historySidebar.classList.add('translate-x-full');
  historyOverlay.classList.add('hidden');
  document.body.style.overflow = '';
}

function showTemporaryMessage(message) {
  // Create temporary success message
  const msgEl = document.createElement('div');
  msgEl.className = 'fixed top-4 right-4 bg-green-500 text-white px-4 py-2 rounded shadow-lg z-50';
  msgEl.textContent = message;
  document.body.appendChild(msgEl);
  setTimeout(() => {
    msgEl.remove();
  }, 3000);
}

// Clear button handler
function clearAll() {
  videoUrl.value = '';
  transcriptBox.value = '';
  summaryBox.value = '';
  // hide metadata
  videoThumb.src = '';
  videoThumb.classList.add('hidden');
  videoTitle.textContent = '';
  videoChannel.textContent = '';
  openLinkBtn.href = '#';
  openLinkBtn.classList.add('hidden');
  videoMeta.classList.add('hidden');
  clearError();
  videoUrl.focus();
}

// Theme functionality
function initTheme() {
  const savedTheme = localStorage.getItem('theme') || 'light';
  document.documentElement.classList.toggle('dark', savedTheme === 'dark');
  updateThemeIcon();
}

function toggleTheme() {
  const isDark = document.documentElement.classList.toggle('dark');
  localStorage.setItem('theme', isDark ? 'dark' : 'light');
  updateThemeIcon();
}

function updateThemeIcon() {
  const icon = themeToggle.querySelector('i');
  const isDark = document.documentElement.classList.contains('dark');
  icon.className = isDark ? 'fa-solid fa-sun' : 'fa-solid fa-moon';
}

// Clipboard functionality
async function copyToClipboard(textarea, type) {
  const text = textarea.value.trim();
  if (!text) {
    showError(`No ${type.toLowerCase()} to copy.`);
    return;
  }
  try {
    await navigator.clipboard.writeText(text);
    showTemporaryMessage(`${type} copied to clipboard!`);
  } catch (err) {
    showError('Failed to copy to clipboard.');
  }
}

// Download functionality
function downloadText(filename, text) {
  if (!text.trim()) {
    showError('No content to download.');
    return;
  }
  const blob = new Blob([text], { type: 'text/plain' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
  showTemporaryMessage('File downloaded!');
}

// Event listeners
form.addEventListener('submit', fetchTranscript);
summarizeBtn.addEventListener('click', summarize);
copyTranscriptBtn.addEventListener('click', () => copyToClipboard(transcriptBox, 'Transcript'));
copySummaryBtn.addEventListener('click', () => copyToClipboard(summaryBox, 'Summary'));
downloadTranscriptBtn.addEventListener('click', () => downloadText('transcript.txt', transcriptBox.value));
downloadSummaryBtn.addEventListener('click', () => downloadText('summary.txt', summaryBox.value));
themeToggle.addEventListener('click', toggleTheme);
clearBtn.addEventListener('click', clearAll);

// History event listeners
historyToggle.addEventListener('click', openHistorySidebar);
closeHistory.addEventListener('click', closeHistorySidebar);
historyOverlay.addEventListener('click', closeHistorySidebar);
clearHistoryBtn.addEventListener('click', clearHistory);

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') {
    closeHistorySidebar();
  }
  if (e.ctrlKey && e.key === 'h') {
    e.preventDefault();
    openHistorySidebar();
  }
});

// Initialize theme
initTheme();

// Initialize history on page load
document.addEventListener('DOMContentLoaded', () => {
  console.log('Page loaded, history items:', transcriptHistory.length);
});