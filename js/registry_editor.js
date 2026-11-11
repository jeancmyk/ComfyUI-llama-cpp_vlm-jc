// ComfyUI-side editor for the llama-cpp OpenAI endpoint's registry.
//
// The "Llama-cpp API Registry" node holds the whole api_models.json document in
// its registry_json text box. This extension keeps that box in sync with the
// file on the server: it fills the box when the node is added, and the button
// re-reads the file (useful after a hand edit, an HTTP write, or another
// browser tab). Saving is the node's own job: running it validates and writes
// through the same path the HTTP editor uses.
//
// The quick-edit form is deliberately a *view* on the box, never a second
// source of truth: it parses the JSON in the box, changes the selected entry's
// fields, and writes the document back. Anything the form does not know about
// (a key added in a later release, a hand-tuned params block) survives the
// round trip untouched.
//
// Silently swallowing errors here would show an empty box ("{}") next to a
// full registry, so every failure is surfaced as a toast.

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const REGISTRY_NODE = "llama_cpp_api_registry";

// The fields worth a control; everything else stays JSON-only.
const FORM_FIELDS = [
    { key: "model", label: "checkpoint", kind: "text" },
    { key: "mmproj", label: "mmproj", kind: "text" },
    { key: "chat_handler", label: "handler", kind: "select", optionsFrom: "chat_handlers" },
    { key: "n_ctx", label: "n_ctx", kind: "int", min: 1024, max: 327680, step: 1024 },
    { key: "cache_type_k", label: "KV k", kind: "select", optionsFrom: "cache_types" },
    { key: "cache_type_v", label: "KV v", kind: "select", optionsFrom: "cache_types" },
    { key: "flash_attn", label: "flash", kind: "select", optionsFrom: "flash_attn" },
    { key: "spec_type", label: "spec", kind: "select", optionsFrom: "spec_types" },
    { key: "spec_draft", label: "draft", kind: "text" },
    { key: "load_mtp", label: "load_mtp", kind: "bool" },
    { key: "free_vram", label: "free_vram", kind: "bool" },
];

let optionsCache = null;

function toast(msg, color = "#333") {
    const el = document.createElement("div");
    el.textContent = msg;
    Object.assign(el.style, {
        position: "fixed", bottom: "30px", right: "30px",
        background: color, color: "#fff",
        padding: "8px 16px", borderRadius: "8px",
        fontSize: "13px", zIndex: "9999",
        boxShadow: "0 2px 8px rgba(0,0,0,.4)",
        transition: "opacity .5s",
    });
    document.body.appendChild(el);
    setTimeout(() => { el.style.opacity = "0"; setTimeout(() => el.remove(), 600); }, 3200);
}

function styleBtn(label, title) {
    const btn = document.createElement("button");
    btn.textContent = label;
    btn.title = title;
    Object.assign(btn.style, {
        padding: "4px 10px", margin: "2px",
        borderRadius: "5px", border: "none",
        background: "#3a7bd5", color: "#fff",
        cursor: "pointer", fontSize: "12px", fontWeight: "bold",
    });
    return btn;
}

function boxOf(node) {
    return node.widgets?.find((w) => w.name === "registry_json");
}

async function pullRegistry(node) {
    const box = boxOf(node);
    if (!box) return;
    const resp = await api.fetchApi("/llama_cpp/api/registry/file");
    if (!resp.ok) {
        const detail = await resp.text().catch(() => "");
        toast(`Registry load failed (${resp.status}): ${detail.slice(0, 140)}`, "#a33");
        return;
    }
    const payload = await resp.json();
    box.value = JSON.stringify(payload.registry, null, 2);
    box.callback?.();
    const n = Object.keys(payload.registry?.models || {}).length;
    const model = payload.loaded_model ? ` | loaded: ${payload.loaded_model}` : "";
    toast(`Registry loaded: ${n} model(s)${model}`, "#2a6");
}

async function loadOptions() {
    if (optionsCache) return optionsCache;
    const resp = await api.fetchApi("/llama_cpp/api/loader_options");
    if (!resp.ok) throw new Error(`options ${resp.status}`);
    optionsCache = await resp.json();
    return optionsCache;
}

function readDoc(box) {
    const parsed = JSON.parse(box.value || "{}");
    parsed.models = parsed.models || {};
    return parsed;
}

function writeDoc(node, box, doc) {
    box.value = JSON.stringify(doc, null, 2);
    box.callback?.();
    node.setDirtyCanvas?.(true);
}

// ---------------------------------------------------------------------------
// quick-edit form
// ---------------------------------------------------------------------------
function buildForm(node) {
    const wrap = document.createElement("div");
    Object.assign(wrap.style, {
        display: "none", flexDirection: "column", gap: "3px",
        padding: "4px", fontSize: "11px", color: "#ccc",
    });

    const head = document.createElement("div");
    Object.assign(head.style, { display: "flex", gap: "4px", alignItems: "center" });
    const entrySelect = document.createElement("select");
    Object.assign(entrySelect.style, { background: "#222", color: "#ddd", border: "1px solid #555", padding: "2px" });
    head.appendChild(document.createElement("span")).textContent = "entry";
    head.appendChild(entrySelect);
    const reloadBtn = styleBtn("↻", "Re-read the entries listed in the box");
    reloadBtn.style.padding = "2px 6px";
    reloadBtn.style.fontSize = "11px";
    head.appendChild(reloadBtn);
    wrap.appendChild(head);

    const grid = document.createElement("div");
    Object.assign(grid.style, {
        display: "grid", gridTemplateColumns: "auto 1fr auto 1fr",
        gap: "3px", alignItems: "center",
    });
    const inputs = {};
    for (const field of FORM_FIELDS) {
        const label = document.createElement("span");
        label.textContent = field.label;
        grid.appendChild(label);
        let control;
        if (field.kind === "select") {
            control = document.createElement("select");
        } else if (field.kind === "int") {
            control = document.createElement("input");
            control.type = "number";
            control.min = field.min; control.max = field.max; control.step = field.step;
        } else if (field.kind === "bool") {
            control = document.createElement("input");
            control.type = "checkbox";
        } else {
            control = document.createElement("input");
            control.type = "text";
        }
        Object.assign(control.style, {
            background: "#222", color: "#ddd", border: "1px solid #555",
            padding: "2px 4px", fontSize: "11px", minWidth: "60px",
        });
        inputs[field.key] = { control, field };
        grid.appendChild(control);
    }
    wrap.appendChild(grid);

    const actions = document.createElement("div");
    Object.assign(actions.style, { display: "flex", gap: "4px" });
    const applyBtn = styleBtn("Apply to selected entry", "Write these values into the entry in the JSON box");
    actions.appendChild(applyBtn);
    wrap.appendChild(actions);

    const selectedId = () => entrySelect.value;

    const fillEntryOptions = () => {
        const box = boxOf(node);
        let doc;
        try {
            doc = readDoc(box);
        } catch (err) {
            toast(`Box is not valid JSON: ${err}`, "#a33");
            return;
        }
        const ids = Object.keys(doc.models);
        const previous = entrySelect.value;
        entrySelect.innerHTML = "";
        for (const id of ids) {
            const opt = document.createElement("option");
            opt.value = id;
            opt.textContent = id;
            entrySelect.appendChild(opt);
        }
        if (ids.includes(previous)) entrySelect.value = previous;
        loadIntoForm(doc);
    };

    async function fillSelects() {
        let options;
        try {
            options = await loadOptions();
        } catch (err) {
            // A missing route means an older server side; the form still works
            // for text/int/bool fields, so degrade instead of failing.
            console.warn("[llama-cpp_vlm] registry options unavailable:", err);
            return;
        }
        for (const { control, field } of Object.values(inputs)) {
            if (field.kind !== "select") continue;
            const list = options[field.optionsFrom] || [];
            control.innerHTML = "";
            for (const value of list) {
                const opt = document.createElement("option");
                opt.value = value;
                opt.textContent = value;
                control.appendChild(opt);
            }
        }
    }

    const loadIntoForm = (doc) => {
        const entry = doc.models?.[selectedId()];
        if (!entry) return;
        for (const { control, field } of Object.values(inputs)) {
            const value = entry[field.key];
            if (field.kind === "bool") control.checked = Boolean(value);
            else if (value === undefined || value === null) control.value = "";
            else control.value = value;
            // Keep an out-of-range current value visible instead of silently
            // replacing it with the first option.
            if (field.kind === "select" && value !== undefined && value !== null) {
                if (![...control.options].some((o) => o.value === String(value))) {
                    const opt = document.createElement("option");
                    opt.value = String(value);
                    opt.textContent = `${value} (current)`;
                    control.appendChild(opt);
                }
                control.value = String(value);
            }
        }
    };

    applyBtn.onclick = () => {
        const box = boxOf(node);
        let doc;
        try {
            doc = readDoc(box);
        } catch (err) {
            toast(`Box is not valid JSON: ${err}`, "#a33");
            return;
        }
        const id = selectedId();
        const entry = doc.models?.[id];
        if (!entry) {
            toast(`No entry named ${id}`, "#a33");
            return;
        }
        for (const { control, field } of Object.values(inputs)) {
            if (field.kind === "bool") entry[field.key] = control.checked;
            else if (field.kind === "int") {
                const num = parseInt(control.value, 10);
                if (!Number.isFinite(num)) continue;
                entry[field.key] = Math.max(field.min, Math.min(field.max, num));
            } else {
                const text = String(control.value ?? "").trim();
                if (text !== "") entry[field.key] = text;
            }
        }
        writeDoc(node, box, doc);
        toast(`Updated '${id}' in the box - run the node (or press Save) to write it`, "#2a6");
    };

    reloadBtn.onclick = fillEntryOptions;
    entrySelect.onchange = () => {
        try {
            loadIntoForm(readDoc(boxOf(node)));
        } catch (err) {
            toast(`Box is not valid JSON: ${err}`, "#a33");
        }
    };

    fillSelects();
    return {
        wrap,
        show: () => {
            wrap.style.display = "flex";
            fillEntryOptions();
            fillSelects();
            node.setSize(node.computeSize());
        },
        hide: () => {
            wrap.style.display = "none";
            node.setSize(node.computeSize());
        },
    };
}

function addControls(node) {
    // Only on the registry node, and only once.
    if (node.controlsAdded) return;
    node.controlsAdded = true;

    const bar = document.createElement("div");
    Object.assign(bar.style, {
        display: "flex", gap: "4px", flexWrap: "wrap",
        padding: "4px", justifyContent: "center",
    });

    const load = styleBtn("Load from server", "Read api_models.json into the box (settings, models, current clamp knobs)");
    load.onclick = () => { pullRegistry(node).catch((e) => toast(`Load failed: ${e}`, "#a33")); };
    bar.appendChild(load);

    const form = buildForm(node);

    const edit = styleBtn("Quick edit", "Edit one entry's common fields instead of typing JSON");
    let open = false;
    edit.onclick = () => {
        open = !open;
        open ? form.show() : form.hide();
    };
    bar.appendChild(edit);

    const create = styleBtn("New entry", "Insert a freshly defaulted entry into the box");
    create.onclick = async () => {
        const box = boxOf(node);
        let doc, options;
        try {
            doc = readDoc(box);
        } catch (err) {
            toast(`Box is not valid JSON: ${err}`, "#a33");
            return;
        }
        try {
            options = await loadOptions();
        } catch (err) {
            toast(`Could not read the defaults: ${err}`, "#a33");
            return;
        }
        const suggested = `model-${Object.keys(doc.models).length + 1}`;
        const id = window.prompt("Model id (what clients pass as \"model\")", suggested);
        if (!id) return;
        if (doc.models[id]) {
            toast(`'${id}' already exists`, "#a33");
            return;
        }
        const entry = { ...(options.defaults || {}) };
        entry.description = "New entry: set 'model' to a checkpoint under models/LLM.";
        doc.models[id] = entry;
        writeDoc(node, box, doc);
        toast(`Added '${id}' with defaults - fill in 'model', then save`, "#2a6");
        if (!open) {
            open = true;
            form.show();
        }
    };
    bar.appendChild(create);

    const wrap = document.createElement("div");
    wrap.appendChild(bar);
    wrap.appendChild(form.wrap);

    node.addDOMWidget("llama_cpp_registry_controls", "btn_bar", wrap, {
        serialize: false,
        hideOnZoom: false,
    });

    // Fill on first add so the box is never a misleading "{}".
    setTimeout(() => { pullRegistry(node).catch((e) => toast(`Load failed: ${e}`, "#a33")); }, 300);
}

app.registerExtension({
    name: "llama_cpp_vlm.registry_editor",
    async nodeCreated(node) {
        if (node.comfyClass === REGISTRY_NODE || node.constructor?.title === "Llama-cpp API Registry") {
            addControls(node);
        }
    },
});
