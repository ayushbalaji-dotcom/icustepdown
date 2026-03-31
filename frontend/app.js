const startBtn = document.getElementById('startBtn');
const newEncounterBtn = document.getElementById('newEncounterBtn');
const nhsInput = document.getElementById('nhsNumber');
const nhsStatus = document.getElementById('nhsStatus');
const preopSection = document.getElementById('preopSection');
const entrySection = document.getElementById('entrySection');
const scoreSection = document.getElementById('scoreSection');
const savePreopBtn = document.getElementById('savePreopBtn');
const preopStatus = document.getElementById('preopStatus');
const submitBtn = document.getElementById('submitBtn');
const dummyBtn = document.getElementById('dummyBtn');
const submitStatus = document.getElementById('submitStatus');
const scoreBtn = document.getElementById('scoreBtn');
const toast = document.getElementById('toast');
const toastMessage = document.getElementById('toastMessage');
const toastClose = document.getElementById('toastClose');
const trainingFileInput = document.getElementById('trainingFile');
const trainingModelNameInput = document.getElementById('trainingModelName');
const trainModelBtn = document.getElementById('trainModelBtn');
const trainingStatus = document.getElementById('trainingStatus');
const modelStatus = document.getElementById('modelStatus');
const modelMetrics = document.getElementById('modelMetrics');

const trafficLight = document.getElementById('trafficLight');
const iriValue = document.getElementById('iriValue');
const trendValue = document.getElementById('trendValue');
const limitingValue = document.getElementById('limitingValue');
const signalsValue = document.getElementById('signalsValue');
const domainList = document.getElementById('domainList');

let currentNhs = null;

function pad(n) { return n < 10 ? `0${n}` : n; }

function setDefaultTimestamp() {
  // Manual entry only; leave timestamp empty.
  document.getElementById('timestamp').value = '';
}

function setPreopReady(isReady) {
  if (isReady) {
    entrySection.classList.remove('hidden');
  } else {
    entrySection.classList.add('hidden');
  }
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

async function refreshModelStatus() {
  const res = await fetch('/api/model-status');
  const data = await res.json();
  const model = data.model || {};
  if (data.status === 'ok' && model.available) {
    const source = model.source === 'explicit' ? 'Manual model' : 'Active trained model';
    const name = model.model_path ? model.model_path.split('/').pop() : 'model';
    modelStatus.textContent = `${source}: ${name}`;
    const metrics = model.metrics || {};
    modelMetrics.textContent = `Calibration ${metrics.calibration_method || 'n/a'} | train rows ${metrics.train_rows ?? '--'} | test rows ${metrics.test_rows ?? '--'}`;
  } else {
    modelStatus.textContent = 'No model loaded. Train a workbook or configure a model path before relying on ML scoring.';
    modelMetrics.textContent = '';
  }
}

async function trainModel() {
  const file = trainingFileInput.files && trainingFileInput.files[0];
  if (!file) {
    trainingStatus.textContent = 'Select an .xlsx workbook first.';
    return;
  }

  trainingStatus.textContent = 'Training model...';
  trainModelBtn.disabled = true;
  const form = new FormData();
  form.append('training_file', file);
  const modelName = String(trainingModelNameInput.value || '').trim();
  if (modelName) {
    form.append('model_name', modelName);
  }

  try {
    const res = await fetch('/api/train', { method: 'POST', body: form });
    const data = await res.json();
    if (data.status !== 'ok') {
      trainingStatus.textContent = data.error || 'Training failed.';
      return;
    }
    const summary = data.summary || {};
    trainingStatus.textContent = `Model trained. ${summary.positive_outcomes || 0} adverse and ${summary.negative_outcomes || 0} normal encounters processed.`;
    await refreshModelStatus();
    if (currentNhs) {
      await refreshScore();
    }
  } catch (err) {
    trainingStatus.textContent = 'Training failed.';
  } finally {
    trainModelBtn.disabled = false;
  }
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
    const enc = data.encounter_id ? ` Encounter ${data.encounter_id}.` : '';
    nhsStatus.textContent = `Patient started. Data stored locally.${enc}`;
    preopSection.classList.remove('hidden');
    entrySection.classList.remove('hidden');
    scoreSection.classList.remove('hidden');
    setDefaultTimestamp();
    clearPreopForm();
    setPreopReady(false);
    await loadPreop();
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
    const enc = data.encounter_id ? ` Encounter ${data.encounter_id}.` : '';
    nhsStatus.textContent = `New encounter started.${enc}`;
    preopSection.classList.remove('hidden');
    entrySection.classList.remove('hidden');
    scoreSection.classList.remove('hidden');
    setDefaultTimestamp();
    clearPreopForm();
    setPreopReady(false);
    await refreshScore();
  } else {
    nhsStatus.textContent = data.error || 'Unable to start new encounter.';
  }
}

function clearPreopForm() {
  const ids = ['preop_age', 'preop_bmi', 'preop_frailty', 'preop_renal', 'preop_lv', 'preop_diabetes'];
  ids.forEach(id => {
    const el = document.getElementById(id);
    if (el) el.value = '';
  });
  preopStatus.textContent = '';
}

function fillPreopForm(preop) {
  if (!preop) return;
  const map = {
    preop_age: preop.age_years,
    preop_bmi: preop.bmi,
    preop_frailty: preop.frailty_score,
    preop_renal: preop.renal_function,
    preop_lv: preop.lv_function,
    preop_diabetes: preop.diabetes === 1 || preop.diabetes === true ? 'Yes' : preop.diabetes === 0 ? 'No' : ''
  };
  Object.entries(map).forEach(([id, value]) => {
    const el = document.getElementById(id);
    if (!el || value == null) return;
    el.value = value;
  });
}

function collectPreop() {
  const getValue = (id) => {
    const el = document.getElementById(id);
    if (!el) return '';
    return String(el.value || '').trim();
  };
  return {
    age_years: getValue('preop_age'),
    bmi: getValue('preop_bmi'),
    frailty_score: getValue('preop_frailty'),
    renal_function: getValue('preop_renal'),
    lv_function: getValue('preop_lv'),
    diabetes: getValue('preop_diabetes')
  };
}

async function loadPreop() {
  if (!currentNhs) return;
  const res = await fetch(`/api/preop?nhs_number=${encodeURIComponent(currentNhs)}`);
  const data = await res.json();
  if (data.status === 'ok') {
    fillPreopForm(data.preop);
    preopStatus.textContent = 'Pre-op data loaded.';
    setPreopReady(true);
  } else {
    preopStatus.textContent = 'Enter pre-op characteristics to continue.';
    setPreopReady(false);
  }
}

async function savePreop() {
  if (!currentNhs) return;
  const preop = collectPreop();
  const res = await fetch('/api/preop', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ nhs_number: currentNhs, preop })
  });
  const data = await res.json();
  if (data.status === 'ok') {
    preopStatus.textContent = 'Pre-op data saved.';
    setPreopReady(true);
  } else {
    preopStatus.textContent = data.error || 'Unable to save pre-op data.';
  }
}

function collectRow() {
  const fields = [
    'timestamp','MAP','HR','RR','SpO2','FiO2',
    'noradrenaline_mcgkgmin','adrenaline_mcgkgmin','dobutamine_mcgkgmin','milrinone_mcgkgmin',
    'urine_output_ml_30min','chest_drain_ml_30min','lactate','haemoglobin_gL','creatinine_umolL',
    'WCC_10e9L','temperature_C','RASS',
    'oxygen_device','arterial_line_present','central_line_present','insulin_infusion','rhythm','imaging_summary'
  ];
  const row = {};
  fields.forEach(id => {
    const el = document.getElementById(id);
    if (!el) return;
    const value = el.value;
    if (value === '') return;
    row[id] = value;
  });
  if (row.rhythm) {
    row.pacing_active = row.rhythm === 'Pacing required' ? 1 : 0;
  }
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
  trendValue.textContent = dash.trend_label || dash.trend || '--';
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
savePreopBtn.addEventListener('click', savePreop);
submitBtn.addEventListener('click', submitRow);
scoreBtn.addEventListener('click', refreshScore);
trainModelBtn.addEventListener('click', trainModel);
toastClose.addEventListener('click', hideToast);
setDefaultTimestamp();
refreshModelStatus();

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
    rhythm: 'Sinus',
    imaging_summary: 'CXR clear'
  };
  Object.entries(dummy).forEach(([key, value]) => {
    const el = document.getElementById(key);
    if (el) el.value = value;
  });
  showToast('Dummy values filled. Enter timestamp manually.', false);
}

dummyBtn.addEventListener('click', fillDummyValues);
