const ui = {
  turnBadge: document.getElementById("turnBadge"),
  violationBadge: document.getElementById("violationBadge"),
  sceneLabel: document.getElementById("sceneLabel"),
  proseContainer: document.getElementById("proseContainer"),
  choicesContainer: document.getElementById("choicesContainer"),
  characterList: document.getElementById("characterList"),
  locationName: document.getElementById("locationName"),
  locationDesc: document.getElementById("locationDesc"),
  guardValidated: document.getElementById("guardValidated"),
  guardViolations: document.getElementById("guardViolations"),
  guardRepairs: document.getElementById("guardRepairs"),
  guardBlock: document.getElementById("guardBlock"),
  skipBtn: document.getElementById("skipBtn"),
  choiceTemplate: document.getElementById("choiceTemplate"),
};

const appState = {
  sessionId: null,
  gameState: null,
  currentScene: null,
  isLoading: false,
  typewriterToken: null,
  lastViolationCount: 0,
};

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function apiNewGame() {
  const response = await fetch("/api/game/new", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({}),
  });
  if (!response.ok) {
    throw new Error(`Failed to create game (${response.status})`);
  }
  return response.json();
}

async function apiChoose(sessionId, choiceId) {
  const response = await fetch(`/api/game/${sessionId}/choose`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ choice_id: choiceId }),
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.detail || `Choice failed (${response.status})`);
  }
  return response.json();
}

async function apiState(sessionId) {
  const response = await fetch(`/api/game/${sessionId}/state`);
  if (!response.ok) {
    throw new Error(`Failed to fetch state (${response.status})`);
  }
  return response.json();
}

function resolveLocationInfo(scene, gameState) {
  if (!scene || !gameState) return null;
  return gameState.locations.find((loc) => loc.id === scene.location) || null;
}

function renderTopBar(scene, gameState) {
  ui.turnBadge.textContent = `turn ${gameState?.turn ?? 0}`;
  const violations = gameState?.guard_stats?.violations_caught ?? 0;
  ui.violationBadge.textContent = `◆ ${violations} violations`;
  ui.sceneLabel.textContent = scene?.scene_id?.replace("_", " ")?.toUpperCase() || "SCENE";
}

function renderCharacters(gameState) {
  ui.characterList.innerHTML = "";
  const chars = gameState?.characters ?? [];
  for (const character of chars) {
    const item = document.createElement("li");
    const left = document.createElement("span");
    const right = document.createElement("span");
    left.textContent = character.name;
    right.textContent = character.location;
    item.append(left, right);
    ui.characterList.appendChild(item);
  }
}

function renderLocation(scene, gameState) {
  const location = resolveLocationInfo(scene, gameState);
  if (!location) {
    ui.locationName.textContent = "-";
    ui.locationDesc.textContent = "-";
    return;
  }

  ui.locationName.classList.remove("fade-in");
  ui.locationDesc.classList.remove("fade-in");
  void ui.locationName.offsetWidth;
  ui.locationName.textContent = location.name;
  ui.locationDesc.textContent = location.description;
  ui.locationName.classList.add("fade-in");
  ui.locationDesc.classList.add("fade-in");
}

function renderGuard(gameState) {
  const guard = gameState?.guard_stats;
  const validated = guard?.proposals_generated ?? 0;
  const violations = guard?.violations_caught ?? 0;
  const repairs = guard?.repairs_applied ?? 0;

  ui.guardValidated.textContent = String(validated);
  ui.guardViolations.textContent = String(violations);
  ui.guardRepairs.textContent = String(repairs);

  if (violations > appState.lastViolationCount) {
    ui.guardBlock.classList.remove("pulse");
    void ui.guardBlock.offsetWidth;
    ui.guardBlock.classList.add("pulse");
    setTimeout(() => ui.guardBlock.classList.remove("pulse"), 700);
  }

  appState.lastViolationCount = violations;
}

async function typewriterReveal(text) {
  if (appState.typewriterToken) {
    appState.typewriterToken.cancelled = true;
  }

  const token = { cancelled: false };
  appState.typewriterToken = token;

  ui.proseContainer.classList.remove("fade-in");
  ui.proseContainer.textContent = "";
  void ui.proseContainer.offsetWidth;
  ui.proseContainer.classList.add("fade-in");

  const delay = text.length > 1600 ? 8 : 18;
  for (let i = 0; i < text.length; i += 1) {
    if (token.cancelled) {
      ui.proseContainer.textContent = text;
      return;
    }
    ui.proseContainer.textContent += text[i];
    await sleep(delay);
  }
}

function setLoading(loading) {
  appState.isLoading = loading;
  const buttons = ui.choicesContainer.querySelectorAll("button");
  buttons.forEach((btn) => {
    btn.disabled = loading;
  });
}

function renderChoices(scene) {
  ui.choicesContainer.innerHTML = "";
  const choices = scene?.choices ?? [];

  if (!choices.length) {
    const done = document.createElement("p");
    done.className = "error-text";
    done.textContent = "The night has reached its end. Start a new game to explore again.";
    ui.choicesContainer.appendChild(done);
    return;
  }

  choices.forEach((choice, idx) => {
    const fragment = ui.choiceTemplate.content.cloneNode(true);
    const button = fragment.querySelector("button");
    const label = fragment.querySelector(".choice-label");
    const desc = fragment.querySelector(".choice-desc");

    button.classList.add("choice-enter");
    button.style.animationDelay = `${idx * 80}ms`;
    label.textContent = `▸ ${choice.label}`;
    desc.textContent = choice.description;

    button.addEventListener("click", async () => {
      await onChoiceSelected(choice.id);
    });

    ui.choicesContainer.appendChild(fragment);
  });
}

async function renderScene(scene, gameState) {
  renderTopBar(scene, gameState);
  renderCharacters(gameState);
  renderLocation(scene, gameState);
  renderGuard(gameState);
  renderChoices(scene);
  await typewriterReveal(scene?.prose ?? "");
}

async function onChoiceSelected(choiceId) {
  if (appState.isLoading || !appState.sessionId) {
    return;
  }

  setLoading(true);
  ui.proseContainer.classList.remove("scene-transition");
  void ui.proseContainer.offsetWidth;
  ui.proseContainer.classList.add("scene-transition");

  try {
    const scene = await apiChoose(appState.sessionId, choiceId);
    const state = await apiState(appState.sessionId);

    appState.currentScene = scene;
    appState.gameState = state;

    await renderScene(scene, state);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    const p = document.createElement("p");
    p.className = "error-text";
    p.textContent = `Scene transition failed: ${message}`;
    ui.choicesContainer.prepend(p);
  } finally {
    setLoading(false);
  }
}

ui.skipBtn.addEventListener("click", () => {
  if (appState.typewriterToken) {
    appState.typewriterToken.cancelled = true;
  }
});

async function bootstrap() {
  try {
    const payload = await apiNewGame();
    appState.sessionId = payload.session_id;
    appState.gameState = payload.state;
    appState.currentScene = payload.scene;
    await renderScene(payload.scene, payload.state);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    ui.proseContainer.textContent = "Unable to start the narrative session.";
    const p = document.createElement("p");
    p.className = "error-text";
    p.textContent = message;
    ui.choicesContainer.appendChild(p);
  }
}

bootstrap();
