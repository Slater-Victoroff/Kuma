import "../src/player/KumaPlayer.js";

const player = document.querySelector<HTMLElement>("#player")!;
const modelSelect = document.querySelector<HTMLSelectElement>("#model-select")!;

modelSelect.addEventListener("change", () => {
  if (!modelSelect.value) return;
  player.setAttribute("src", modelSelect.value);
});

async function populateArtifacts(): Promise<void> {
  let files: string[] = [];
  try {
    const res = await fetch("/api/artifacts");
    files = (await res.json()) as string[];
  } catch {
    // dev server not available or no artifacts dir — leave list empty
  }

  modelSelect.innerHTML = "";

  if (files.length === 0) {
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "(no .iph files found in artifacts/)";
    modelSelect.appendChild(opt);
    return;
  }

  const placeholder = document.createElement("option");
  placeholder.value = "";
  placeholder.textContent = "— select a model —";
  modelSelect.appendChild(placeholder);

  for (const name of files) {
    const opt = document.createElement("option");
    opt.value = `/artifacts/${name}`;
    opt.textContent = name;
    modelSelect.appendChild(opt);
  }
}

void populateArtifacts();
