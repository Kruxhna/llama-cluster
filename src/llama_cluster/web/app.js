/**
 * AeroMesh Web Control Dashboard JavaScript Client.
 * Reactive state management, live 200ms telemetry streaming, dynamic layer slicing,
 * topology editor, and distributed inference test console.
 */

// Application State
const state = {
  cluster: null,
  telemetry: null,
  allocations: {},
  totalLayers: 48,
  isEditingNodeIndex: -1,
  pollTimer: null,
};

const COLOR_CLASSES = [
  'seg-laptop-a',
  'seg-laptop-b',
  'seg-laptop-c',
  'seg-laptop-d',
];

// DOM Elements
const elements = {
  headerModelName: document.getElementById('headerModelName'),
  headerTotalVram: document.getElementById('headerTotalVram'),
  headerActiveNodes: document.getElementById('headerActiveNodes'),
  clusterPulse: document.getElementById('clusterPulse'),
  clusterStatusText: document.getElementById('clusterStatusText'),
  nodeCardsGrid: document.getElementById('nodeCardsGrid'),
  totalLayersCount: document.getElementById('totalLayersCount'),
  allocatedLayersCount: document.getElementById('allocatedLayersCount'),
  maxLayersBudget: document.getElementById('maxLayersBudget'),
  layerBarSegments: document.getElementById('layerBarSegments'),
  nodeSlidersGrid: document.getElementById('nodeSlidersGrid'),
  btnAutoBalance: document.getElementById('btnAutoBalance'),
  btnApplyLayers: document.getElementById('btnApplyLayers'),
  btnQuickStart: document.getElementById('btnQuickStart'),
  cfgClusterName: document.getElementById('cfgClusterName'),
  cfgCoordinatorSelect: document.getElementById('cfgCoordinatorSelect'),
  cfgModelSelect: document.getElementById('cfgModelSelect'),
  nodeTableBody: document.getElementById('nodeTableBody'),
  clusterConfigForm: document.getElementById('clusterConfigForm'),
  btnAddNodeModal: document.getElementById('btnAddNodeModal'),
  nodeModal: document.getElementById('nodeModal'),
  btnCloseModal: document.getElementById('btnCloseModal'),
  btnCancelModal: document.getElementById('btnCancelModal'),
  modalNodeForm: document.getElementById('modalNodeForm'),
  modalTitle: document.getElementById('modalTitle'),
  modalNodeName: document.getElementById('modalNodeName'),
  modalNodeIp: document.getElementById('modalNodeIp'),
  modalNodePort: document.getElementById('modalNodePort'),
  modalNodeGpu: document.getElementById('modalNodeGpu'),
  modalNodeVram: document.getElementById('modalNodeVram'),
  chatForm: document.getElementById('chatForm'),
  chatInput: document.getElementById('chatInput'),
  chatMessages: document.getElementById('chatMessages'),
  btnSendChat: document.getElementById('btnSendChat'),
  inferenceStatus: document.getElementById('inferenceStatus'),
  inferenceSpeed: document.getElementById('inferenceSpeed'),
  inferenceLatency: document.getElementById('inferenceLatency'),
};

// Initialize Dashboard
async function initDashboard() {
  await fetchClusterState();
  await fetchTelemetry();

  // Start telemetry polling loop
  state.pollTimer = setInterval(fetchTelemetry, 1500);

  // Setup Event Listeners
  setupEventListeners();
}

// Fetch Full Cluster Topology & Allocations
async function fetchClusterState() {
  try {
    const res = await fetch('/api/cluster');
    if (!res.ok) throw new Error('Failed to fetch cluster state');
    state.cluster = await res.json();

    state.totalLayers = state.cluster.model_spec?.total_layers || 48;
    state.allocations = { ...state.cluster.allocations };

    renderHeader();
    renderLayerSlicer();
    renderConfigForm();
  } catch (err) {
    console.error('Error loading cluster state:', err);
  }
}

// Fetch Live Hardware Telemetry
async function fetchTelemetry() {
  try {
    const res = await fetch('/api/telemetry');
    if (!res.ok) return;
    state.telemetry = await res.json();
    renderNodeCards();
  } catch (err) {
    console.warn('Telemetry polling error:', err);
  }
}

// Render Top Header Stats
function renderHeader() {
  if (!state.cluster) return;

  const modelName = state.cluster.model_spec?.name || state.cluster.model_spec?.file || 'DeepSeek-R1-14B';
  elements.headerModelName.textContent = modelName;

  const totalVram = state.cluster.nodes?.reduce((acc, n) => acc + (n.usable_vram_gb || 0), 0) || 0;
  elements.headerTotalVram.textContent = `${totalVram.toFixed(1)} GB`;

  const count = state.cluster.nodes?.length || 0;
  elements.headerActiveNodes.textContent = `${count} Nodes`;

  elements.totalLayersCount.textContent = state.totalLayers;
  elements.maxLayersBudget.textContent = state.totalLayers;
}

// Render Telemetry Node Cards
function renderNodeCards() {
  if (!state.telemetry?.nodes || !state.cluster) return;

  elements.nodeCardsGrid.innerHTML = '';

  state.telemetry.nodes.forEach((node) => {
    const isCoord = node.is_coordinator;
    const isOnline = node.online;
    const card = document.createElement('div');
    card.className = `card node-card ${isCoord ? 'is-coordinator' : 'is-worker'}`;

    const tempClass = node.gpu_temp_celsius > 80 ? 'meter-fill-red' : (node.gpu_temp_celsius > 65 ? 'meter-fill-amber' : 'meter-fill-green');
    const assignedLayers = state.allocations[node.name] || 0;

    card.innerHTML = `
      <div class="node-card-header">
        <div class="node-title-group">
          <h3>${node.name}</h3>
          <span class="node-ip-tag">${node.ip}:${node.port}</span>
        </div>
        <span class="node-role-badge ${isCoord ? 'badge-coordinator' : 'badge-worker'}">
          ${isCoord ? 'Coordinator' : 'Worker'}
        </span>
      </div>

      <div class="node-gpu-info">
        <span>🎮 ${node.gpu_model}</span>
        <span class="text-cyan"><strong>${assignedLayers}</strong> layers</span>
      </div>

      <div class="metrics-row">
        <div class="metric-box">
          <div class="metric-label">GPU Temp</div>
          <div class="metric-val ${node.gpu_temp_celsius > 80 ? 'text-red' : ''}">${node.gpu_temp_celsius}°C</div>
          <div class="metric-meter"><div class="meter-fill ${tempClass}" style="width: ${Math.min(100, (node.gpu_temp_celsius/90)*100)}%"></div></div>
        </div>

        <div class="metric-box">
          <div class="metric-label">VRAM Free</div>
          <div class="metric-val text-cyan">${node.vram_free_gb} GB</div>
          <div class="metric-meter"><div class="meter-fill meter-fill-cyan" style="width: ${Math.min(100, (node.vram_free_gb/12)*100)}%"></div></div>
        </div>

        <div class="metric-box">
          <div class="metric-label">CPU Load</div>
          <div class="metric-val text-purple">${node.cpu_utilization_pct}%</div>
          <div class="metric-meter"><div class="meter-fill meter-fill-purple" style="width: ${node.cpu_utilization_pct}%"></div></div>
        </div>

        <div class="metric-box">
          <div class="metric-label">Mesh RTT</div>
          <div class="metric-val ${isOnline ? 'text-green' : 'text-red'}">${isOnline ? node.rtt_ms + ' ms' : 'Offline'}</div>
          <div class="metric-meter"><div class="meter-fill ${isOnline ? 'meter-fill-green' : 'meter-fill-red'}" style="width: ${isOnline ? Math.min(100, (node.rtt_ms/300)*100) : 0}%"></div></div>
        </div>
      </div>

      <div class="node-footer">
        <div class="node-link-status">
          <span class="link-dot ${isOnline ? 'online' : 'offline'}"></span>
          <span>${isOnline ? 'Link Active (P2P Mesh)' : 'Awaiting `aeromesh node`'}</span>
        </div>
        <span>${node.gpu_power_watts} W</span>
      </div>
    `;

    elements.nodeCardsGrid.appendChild(card);
  });
}

// Render Dynamic Layer Slicers & Progress Bar
function renderLayerSlicer() {
  if (!state.cluster?.nodes) return;

  const nodes = state.cluster.nodes;
  const currentSum = Object.values(state.allocations).reduce((a, b) => a + b, 0);

  elements.allocatedLayersCount.textContent = currentSum;
  elements.allocatedLayersCount.className = currentSum === state.totalLayers ? 'text-cyan' : 'text-amber';

  // Render Visual Segment Bar
  elements.layerBarSegments.innerHTML = '';
  nodes.forEach((n, idx) => {
    const layers = state.allocations[n.name] || 0;
    const pct = ((layers / state.totalLayers) * 100).toFixed(1);
    const colorClass = COLOR_CLASSES[idx % COLOR_CLASSES.length];

    if (layers > 0) {
      const seg = document.createElement('div');
      seg.className = `layer-segment ${colorClass}`;
      seg.style.width = `${pct}%`;
      seg.title = `${n.name}: ${layers} layers (${pct}%)`;
      seg.innerHTML = `<span>${n.name}: ${layers}L</span>`;
      elements.layerBarSegments.appendChild(seg);
    }
  });

  // Render Sliders Grid
  elements.nodeSlidersGrid.innerHTML = '';
  nodes.forEach((n, idx) => {
    const layers = state.allocations[n.name] || 0;
    const sliderCard = document.createElement('div');
    sliderCard.className = 'slider-card';

    sliderCard.innerHTML = `
      <div class="slider-header">
        <span class="slider-node-name">${n.name}</span>
        <span class="slider-layer-badge" id="badge_${n.name}">${layers} Layers</span>
      </div>
      <input type="range" class="layer-range-slider" 
             id="slider_${n.name}" 
             min="0" max="${state.totalLayers}" 
             value="${layers}">
      <div class="slider-meta">
        <span>GPU: ${n.gpu_model}</span>
        <span>VRAM: ${n.usable_vram_gb} GB</span>
      </div>
    `;

    elements.nodeSlidersGrid.appendChild(sliderCard);

    // Attach Range Input Event
    const input = sliderCard.querySelector(`#slider_${n.name}`);
    input.addEventListener('input', (e) => {
      const val = parseInt(e.target.value, 10);
      state.allocations[n.name] = val;
      document.getElementById(`badge_${n.name}`).textContent = `${val} Layers`;
      updateLayerBarLive();
    });
  });
}

// Live update of layer bar without full re-render
function updateLayerBarLive() {
  const currentSum = Object.values(state.allocations).reduce((a, b) => a + b, 0);
  elements.allocatedLayersCount.textContent = currentSum;
  elements.allocatedLayersCount.className = currentSum === state.totalLayers ? 'text-cyan' : 'text-amber';

  const nodes = state.cluster.nodes;
  elements.layerBarSegments.innerHTML = '';
  nodes.forEach((n, idx) => {
    const layers = state.allocations[n.name] || 0;
    const pct = ((layers / state.totalLayers) * 100).toFixed(1);
    const colorClass = COLOR_CLASSES[idx % COLOR_CLASSES.length];

    if (layers > 0) {
      const seg = document.createElement('div');
      seg.className = `layer-segment ${colorClass}`;
      seg.style.width = `${pct}%`;
      seg.title = `${n.name}: ${layers} layers (${pct}%)`;
      seg.innerHTML = `<span>${n.name}: ${layers}L</span>`;
      elements.layerBarSegments.appendChild(seg);
    }
  });
}

// Render Config Form (Coordinator, Models, Table)
function renderConfigForm() {
  if (!state.cluster) return;

  elements.cfgClusterName.value = state.cluster.cluster_name || 'aeromesh-tailscale-mesh';

  // Populate Coordinator Select
  elements.cfgCoordinatorSelect.innerHTML = '';
  state.cluster.nodes?.forEach((n) => {
    const opt = document.createElement('option');
    opt.value = n.name;
    opt.textContent = `${n.name} (${n.ip})`;
    if (n.name === state.cluster.coordinator?.node_id || n.is_stable_coordinator) {
      opt.selected = true;
    }
    elements.cfgCoordinatorSelect.appendChild(opt);
  });

  // Populate Models Select
  elements.cfgModelSelect.innerHTML = '';
  const currentModelFile = state.cluster.model_spec?.file;
  (state.cluster.available_models || []).forEach((m) => {
    const opt = document.createElement('option');
    opt.value = m;
    opt.textContent = m;
    if (m === currentModelFile) opt.selected = true;
    elements.cfgModelSelect.appendChild(opt);
  });

  // Render Table Rows
  renderNodeTable();
}

// Render Node Table
function renderNodeTable() {
  elements.nodeTableBody.innerHTML = '';
  state.cluster?.nodes?.forEach((n, idx) => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td><strong>${n.name}</strong> ${n.is_stable_coordinator ? '<span class="text-purple">(Coord)</span>' : ''}</td>
      <td class="node-ip-tag">${n.ip}</td>
      <td>${n.rpc_port}</td>
      <td>${n.usable_vram_gb} GB</td>
      <td>
        <button type="button" class="btn btn-xs btn-outline" onclick="editNode(${idx})">Edit</button>
        <button type="button" class="btn btn-xs btn-secondary text-red" onclick="deleteNode(${idx})">Remove</button>
      </td>
    `;
    elements.nodeTableBody.appendChild(tr);
  });
}

// Setup Dashboard Event Listeners
function setupEventListeners() {
  // Auto-Balance (PuLP ILP)
  elements.btnAutoBalance.addEventListener('click', async () => {
    elements.btnAutoBalance.textContent = '⏳ Solving ILP...';
    try {
      const res = await fetch('/api/rebalance', { method: 'POST' });
      const result = await res.json();
      state.allocations = result.allocations || {};
      renderLayerSlicer();
      elements.btnAutoBalance.innerHTML = '<span>🧠</span> Auto-Balance (PuLP ILP)';
    } catch (e) {
      alert('Failed to solve layer balance: ' + e);
      elements.btnAutoBalance.innerHTML = '<span>🧠</span> Auto-Balance (PuLP ILP)';
    }
  });

  // Apply Layer Slices
  elements.btnApplyLayers.addEventListener('click', async () => {
    elements.btnApplyLayers.textContent = 'Applying...';
    try {
      const res = await fetch('/api/apply-layers', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ allocations: state.allocations }),
      });
      const data = await res.json();
      elements.btnApplyLayers.innerHTML = '<span>✅</span> Layers Applied!';
      setTimeout(() => {
        elements.btnApplyLayers.innerHTML = '<span>💾</span> Apply Layer Slices';
      }, 2000);
    } catch (e) {
      alert('Error applying layer slices: ' + e);
      elements.btnApplyLayers.innerHTML = '<span>💾</span> Apply Layer Slices';
    }
  });

  // Quick Start Cluster
  elements.btnQuickStart.addEventListener('click', async () => {
    const isRunning = elements.btnQuickStart.textContent.includes('Stop');
    if (isRunning) {
      await fetch('/api/cluster/stop', { method: 'POST' });
      elements.btnQuickStart.innerHTML = '<span>🚀</span> Start Cluster';
      elements.clusterStatusText.textContent = 'Mesh Standby';
    } else {
      elements.btnQuickStart.textContent = 'Starting...';
      const res = await fetch('/api/cluster/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: elements.cfgModelSelect.value }),
      });
      const data = await res.json();
      elements.btnQuickStart.innerHTML = '<span>⏹️</span> Stop Cluster';
      elements.clusterStatusText.textContent = 'Inference Active';
    }
  });

  // Save Cluster Config
  elements.clusterConfigForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const coordName = elements.cfgCoordinatorSelect.value;
    const selectedModel = elements.cfgModelSelect.value;

    const updatedNodes = state.cluster.nodes.map((n) => ({
      ...n,
      is_stable_coordinator: n.name === coordName,
    }));

    const coordNode = updatedNodes.find((n) => n.name === coordName);

    const payload = {
      cluster_name: elements.cfgClusterName.value,
      coordinator: {
        node_id: coordName,
        host: coordNode ? coordNode.ip : '127.0.0.1',
        control_port: 8080,
      },
      model_spec: {
        ...state.cluster.model_spec,
        name: selectedModel.replace('.gguf', ''),
        file: selectedModel,
      },
      nodes: updatedNodes,
    };

    try {
      const res = await fetch('/api/config/update', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.error || 'Server error ' + res.status);
      }
      alert('Configuration updated & saved to cluster.yaml successfully!');
      await fetchClusterState();
    } catch (err) {
      alert('Failed to save config: ' + err);
    }
  });

  // Add Node Modal
  elements.btnAddNodeModal.addEventListener('click', () => {
    state.isEditingNodeIndex = -1;
    elements.modalTitle.textContent = 'Add New Cluster Node';
    elements.modalNodeForm.reset();
    elements.nodeModal.classList.add('active');
  });

  elements.btnCloseModal.addEventListener('click', () => elements.nodeModal.classList.remove('active'));
  elements.btnCancelModal.addEventListener('click', () => elements.nodeModal.classList.remove('active'));

  // Save Node from Modal
  elements.modalNodeForm.addEventListener('submit', (e) => {
    e.preventDefault();
    const nodeData = {
      name: elements.modalNodeName.value.trim(),
      role: 'Compute Worker',
      ip: elements.modalNodeIp.value.trim(),
      rpc_port: parseInt(elements.modalNodePort.value, 10),
      gpu_model: elements.modalNodeGpu.value.trim(),
      usable_vram_gb: parseFloat(elements.modalNodeVram.value),
      compute_tflops: 15.0,
      is_stable_coordinator: false,
    };

    if (state.isEditingNodeIndex >= 0) {
      state.cluster.nodes[state.isEditingNodeIndex] = nodeData;
    } else {
      state.cluster.nodes.push(nodeData);
    }

    elements.nodeModal.classList.remove('active');
    renderConfigForm();
    renderLayerSlicer();
  });

  // Chat Form Test Generation
  elements.chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const prompt = elements.chatInput.value.trim();
    if (!prompt) return;

    appendChatMessage('user', prompt);
    elements.chatInput.value = '';
    elements.inferenceStatus.textContent = 'Generating...';
    elements.inferenceStatus.className = 'text-amber';

    const startTime = performance.now();
    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt, max_tokens: 60 }),
      });
      const data = await res.json();
      const endTime = performance.now();
      const elapsedMs = endTime - startTime;

      if (data.error) {
        appendChatMessage('system', `Error: ${data.error}`);
        elements.inferenceStatus.textContent = 'Error';
        elements.inferenceStatus.className = 'text-red';
      } else {
        const choice = data.choices?.[0];
        const content = choice?.message?.content || choice?.message?.reasoning_content || JSON.stringify(data);
        appendChatMessage('assistant', content);

        const tokens = data.usage?.completion_tokens || 30;
        const tokSec = ((tokens / (elapsedMs / 1000))).toFixed(1);

        elements.inferenceSpeed.textContent = `${tokSec} tok/s`;
        elements.inferenceLatency.textContent = `${Math.round(elapsedMs)} ms`;
        elements.inferenceStatus.textContent = 'Ready';
        elements.inferenceStatus.className = 'text-green';
      }
    } catch (err) {
      appendChatMessage('system', `Request failed: ${err}`);
      elements.inferenceStatus.textContent = 'Offline';
      elements.inferenceStatus.className = 'text-red';
    }
  });
}

function appendChatMessage(role, content) {
  const msg = document.createElement('div');
  msg.className = `chat-msg msg-${role}`;
  msg.innerHTML = `
    <div class="msg-author">${role === 'user' ? 'You' : (role === 'assistant' ? 'AeroMesh Mesh' : 'System')}</div>
    <div class="msg-content">${escapeHtml(content)}</div>
  `;
  elements.chatMessages.appendChild(msg);
  elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
}

function escapeHtml(str) {
  return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

// Global Edit/Delete Helpers for Table Buttons
window.editNode = function (index) {
  state.isEditingNodeIndex = index;
  const n = state.cluster.nodes[index];
  elements.modalTitle.textContent = `Edit Node: ${n.name}`;
  elements.modalNodeName.value = n.name;
  elements.modalNodeIp.value = n.ip;
  elements.modalNodePort.value = n.rpc_port;
  elements.modalNodeGpu.value = n.gpu_model;
  elements.modalNodeVram.value = n.usable_vram_gb;
  elements.nodeModal.classList.add('active');
};

window.deleteNode = function (index) {
  if (confirm(`Remove node ${state.cluster.nodes[index].name} from cluster?`)) {
    state.cluster.nodes.splice(index, 1);
    renderConfigForm();
    renderLayerSlicer();
  }
};

// Start application
window.addEventListener('DOMContentLoaded', initDashboard);
