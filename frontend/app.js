const startBtn = document.getElementById('startBtn');
const newEncounterBtn = document.getElementById('newEncounterBtn');
const nhsInput = document.getElementById('nhsNumber');
const nhsStatus = document.getElementById('nhsStatus');
const entrySection = document.getElementById('entrySection');
const scoreSection = document.getElementById('scoreSection');
const submitBtn = document.getElementById('submitBtn');
const dummyBtn = document.getElementById('dummyBtn');
const submitStatus = document.getElementById('submitStatus');
const scoreBtn = document.getElementById('scoreBtn');
const toast = document.getElementById('toast');
const toastMessage = document.getElementById('toastMessage');
const toastClose = document.getElementById('toastClose');

const trafficLight = document.getElementById('trafficLight');
const iriValue = document.getElementById('iriValue');
const trendValue = document.getElementById('trendValue');
const limitingValue = document.getElementById('limitingValue');
const signalsValue = document.getElementById('signalsValue');
const domainList = document.getElementById('domainList');

let currentNhs = null;
const storedNhs = localStorage.getItem('icu_stepdown_nhs');
if (storedNhs) {
  currentNhs = storedNhs;
}

function pad(n) { return n < 10 ? `0${n}` : n; }

function setDefaultTimestamp() {
  // Manual entry only; leave timestamp empty.
  document.getElementById('timestamp').value = '';
}

function trafficClass(value) {
  const v = String(value || '').toUpperCase();
  if (v === 'GREEN') return 'green';
  if (v === 'AMBER') return 'amber';
  return 'red';
}

function renderDomains(domainFlags) {
  domainList.innerHTML = '';
  const domains = [
    'Respiratory',
    'Neurological',
    'Cardiovascular',
    'Surgical',
    'Haemodynamics',
    'Imaging'
  ];
  domains.forEach(name => {
    const item = document.createElement('div');
    item.className = 'domain-item';
    const label = document.createElement('span');
    label.textContent = name;
    const badge = document.createElement('span');
    badge.className = 'badge';
    const value = domainFlags && name in domainFlags ? domainFlags[name] : false;
    if (value === 'Not assessed by model') {
      badge.textContent = 'Info';
    } else if (value) {
      badge.textContent = 'Limiting';
      badge.classList.add('on');
    } else {
      badge.textContent = 'OK';
    }
    item.appendChild(label);
    item.appendChild(badge);
    domainList.appendChild(item);
  });
}

async function startPatient() {
  const nhs = nhsInput.value.trim();
  if (!nhs) {
    nhsStatus.textContent = 'Enter an NHS number first.';
    return;
  }
  const res = await fetch('/api/start', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ nhs_number: nhs })
  });
  const data = await res.json();
  if (data.status === 'ok') {
    currentNhs = nhs;
    localStorage.setItem('icu_stepdown_nhs', nhs);
    const enc = data.encounter_id ? ` Encounter ${data.encounter_id}.` : '';
    nhsStatus.textContent = `Patient started. Data stored locally.${enc}`;
    entrySection.classList.remove('hidden');
    scoreSection.classList.remove('hidden');
    setDefaultTimestamp();
    await refreshScore();
  } else {
    nhsStatus.textContent = data.error || 'Unable to start.';
  }
}

async function startNewEncounter() {
  const nhs = nhsInput.value.trim();
  if (!nhs) {
    nhsStatus.textContent = 'Enter an NHS number first.';
    return;
  }
  const res = await fetch('/api/start', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ nhs_number: nhs, force_new: true })
  });
  const data = await res.json();
  if (data.status === 'ok') {
    currentNhs = nhs;
    localStorage.setItem('icu_stepdown_nhs', nhs);
    const enc = data.encounter_id ? ` Encounter ${data.encounter_id}.` : '';
    nhsStatus.textContent = `New encounter started.${enc}`;
    entrySection.classList.remove('hidden');
    scoreSection.classList.remove('hidden');
    setDefaultTimestamp();
    await refreshScore();
  } else {
    nhsStatus.textContent = data.error || 'Unable to start new encounter.';
  }
}

function collectRow() {
  const fields = [
    'timestamp','MAP','HR','RR','SpO2','FiO2',
    'noradrenaline_mcgkgmin','adrenaline_mcgkgmin','dobutamine_mcgkgmin','milrinone_mcgkgmin',
    'urine_output_ml_30min','chest_drain_ml_30min','lactate','haemoglobin_gL','creatinine_umolL',
    'WCC_10e9L','temperature_C','RASS',
    'oxygen_device','arterial_line_present','central_line_present','insulin_infusion','pacing_active','imaging_summary'
  ];
  const row = {};
  fields.forEach(id => {
    const el = document.getElementById(id);
    if (!el) return;
    const value = el.value;
    if (value === '') return;
    row[id] = value;
  });
  return row;
}

function showToast(message, isError = false) {
  toastMessage.textContent = message;
  toast.classList.remove('hidden');
  toast.classList.toggle('error', isError);
}

function hideToast() {
  toast.classList.add('hidden');
  toast.classList.remove('error');
}

async function submitRow() {
  if (!currentNhs) return;
  const row = collectRow();
  if (!row.timestamp) {
    submitStatus.textContent = 'Timestamp required.';
    showToast('Timestamp is required before saving.', true);
    return;
  }
  const res = await fetch('/api/append', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ nhs_number: currentNhs, row })
  });
  const data = await res.json();
  if (data.status === 'ok') {
    submitStatus.textContent = 'Saved hourly data.';
    showToast('Hourly data saved to the database.', false);
    await refreshScore();
  } else {
    submitStatus.textContent = data.error || 'Save failed.';
    showToast(data.error || 'Save failed.', true);
  }
}

async function refreshScore() {
  if (!currentNhs) return;
  const res = await fetch(`/api/score?nhs_number=${encodeURIComponent(currentNhs)}`);
  const data = await res.json();
  if (data.status === 'insufficient_data') {
    trafficLight.textContent = 'WAIT';
    trafficLight.className = 'traffic red';
    iriValue.textContent = '--';
    trendValue.textContent = '--';
    const span = data.hours != null ? Number(data.hours).toFixed(2) : '--';
    limitingValue.textContent = 'Need 4h of data';
    signalsValue.textContent = `No score until 4 hours of data. Rows: ${data.row_count || 0}. Span: ${span}h (min ${data.min_timestamp || '--'}, max ${data.max_timestamp || '--'}).`;
    renderDomains(data.domain_flags || {});
    return;
  }
  if (data.status !== 'ok') {
    trafficLight.textContent = 'RED';
    trafficLight.className = 'traffic red';
    signalsValue.textContent = data.message || 'Score unavailable.';
    return;
  }
  const dash = data.dashboard || {};
  trafficLight.textContent = dash.traffic_light || 'RED';
  trafficLight.className = `traffic ${trafficClass(dash.traffic_light)}`;
  iriValue.textContent = dash.IRI != null ? Number(dash.IRI).toFixed(1) : '--';
  trendValue.textContent = dash.trend || '--';
  limitingValue.textContent = dash.limiting_factor || '--';
  if (data.warning === 'no_model_loaded_fail_closed') {
    signalsValue.textContent = 'No model loaded: showing fail-closed RED. Load a model to enable scoring.';
  } else {
    signalsValue.textContent = dash.signals || 'Insufficient data to explain';
  }
  renderDomains(data.domain_flags || {});
}

startBtn.addEventListener('click', startPatient);
newEncounterBtn.addEventListener('click', startNewEncounter);
submitBtn.addEventListener('click', submitRow);
scoreBtn.addEventListener('click', refreshScore);
toastClose.addEventListener('click', hideToast);
setDefaultTimestamp();

function fillDummyValues() {
  const dummy = {
    MAP: 72,
    HR: 88,
    RR: 18,
    SpO2: 96,
    FiO2: 0.4,
    noradrenaline_mcgkgmin: 0,
    adrenaline_mcgkgmin: 0,
    dobutamine_mcgkgmin: 0,
    milrinone_mcgkgmin: 0,
    urine_output_ml_30min: 60,
    chest_drain_ml_30min: 5,
    lactate: 1.2,
    haemoglobin_gL: 110,
    creatinine_umolL: 95,
    WCC_10e9L: 8.5,
    temperature_C: 37.1,
    RASS: 0,
    oxygen_device: 'NC',
    arterial_line_present: 1,
    central_line_present: 1,
    insulin_infusion: 0,
    pacing_active: 0,
    imaging_summary: 'CXR clear'
  };
  Object.entries(dummy).forEach(([key, value]) => {
    const el = document.getElementById(key);
    if (el) el.value = value;
  });
  showToast('Dummy values filled. Enter timestamp manually.', false);
}

dummyBtn.addEventListener('click', fillDummyValues);

if (currentNhs) {
  nhsInput.value = currentNhs;
  entrySection.classList.remove('hidden');
  scoreSection.classList.remove('hidden');
  refreshScore();
}
