// ComfyUI-side ergonomics for the "Llama-cpp Model Loader" node.
//
// Three things, none of which the server can do for us:
//
//  1. Section toggles. The loader grew a lot of knobs (attention, MoE offload,
//     batch sizes, speculation). Grouping them keeps the node readable, and
//     hiding a group is the only way to keep ~22 widgets on one node sane.
//
//  2. Conditional visibility. Half the knobs are inert for most setups: the
//     draft widgets mean nothing unless spec_type is set, the MoE widgets mean
//     nothing unless cpu_moe is on. Showing them always invites "why didn't
//     that change anything" - so they appear when they matter, and the KV/
//     attention widgets say when a pairing is illegal.
//
//  3. Applied settings. The loader coerces KV types the cache cannot hold and
//     resolves flash_attn=auto to a real on/off; until now that was console-only.
//     GET /llama_cpp/api/loader_status hands the same numbers back per node id,
//     and this renders them on the node.
//
// Nothing here changes what is executed: widgets are hidden, never removed, so
// saved workflows and widget order are untouched.

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const LOADER = "llama_cpp_model_loader";

const SECTIONS = {
    core: {
        label: "Model",
        hint: "Checkpoint, projector, template, context size and image-token budget",
        widgets: ["model", "mmproj", "chat_handler", "n_ctx", "skip_mmproj",
                  "image_min_tokens", "image_max_tokens"],
    },
    memory: {
        label: "VRAM",
        hint: "Layer offload, eviction policy, KV cache type, flash attention",
        widgets: ["vram_limit", "free_vram", "enforce_vram", "cache_type_k", "cache_type_v", "flash_attn"],
    },
    moe: {
        label: "MoE",
        hint: "Keep expert weights in system RAM (mixture-of-experts checkpoints)",
        widgets: ["cpu_moe", "n_cpu_moe", "moe_cache_mb"],
    },
    batch: {
        label: "Batch",
        hint: "Prefill batch sizes (logical / physical)",
        widgets: ["n_batch", "n_ubatch"],
    },
    spec: {
        label: "Spec",
        hint: "Speculative decoding: MTP, DSpark, DFlash, EAGLE3, n-gram",
        widgets: ["load_mtp", "spec_type", "spec_draft", "spec_draft_n_max"],
    },
};

// Hidden groups that hold no interesting value are worth folding by default;
// everything else starts visible so nothing is quietly out of sight.
const COLLAPSED_BY_DEFAULT = ["moe", "batch"];

// Widgets that only mean something when another widget is set a certain way.
function visibilityFor(widgets) {
    const valueOf = (name) => widgets[name]?.value;
    const show = {};
    const spec = String(valueOf("spec_type") ?? "none").toLowerCase();
    show.spec_draft = spec !== "none";
    show.spec_draft_n_max = spec !== "none";
    // The legacy toggle only matters when no speculative engine is selected.
    // It stays visible in that case even while off, or there would be no way to
    // turn it on.
    show.load_mtp = spec === "none";

    const moe = Boolean(valueOf("cpu_moe"));
    show.n_cpu_moe = moe;
    show.moe_cache_mb = true;   // the expert cache is independent of cpu_moe
    return show;
}

// quantized KV needs flash attention; llama.cpp turns it on by itself and the
// loader reports that, but an explicit "off" + quantized cache is a warning.
function kvWarning(widgets) {
    const fa = String(widgets.flash_attn?.value ?? "auto").toLowerCase();
    const k = String(widgets.cache_type_k?.value ?? "f16").toLowerCase();
    const v = String(widgets.cache_type_v?.value ?? "f16").toLowerCase();
    const quantized = (t) => t !== "f16" && t !== "f32" && t !== "bf16";
    if (fa === "off" && (quantized(k) || quantized(v))) {
        return "quantized KV forces flash attention on - 'off' will be overridden";
    }
    return null;
}

function setWidgetVisible(widget, visible) {
    // litegraph skips widgets flagged hidden and numVisibleWidgets() agrees, so
    // hiding really does shrink the node after a computeSize().
    widget.options = widget.options || {};
    widget.options.hidden = !visible;
}

async function fetchStatus() {
    const resp = await api.fetchApi("/llama_cpp/api/loader_status");
    if (!resp.ok) throw new Error(`status ${resp.status}`);
    return resp.json();
}

function formatStatus(entry, backendLine) {
    if (!entry) return "";
    const bits = [];
    if (entry.cache_applied) {
        bits.push(`KV ${entry.cache_applied}`);
        if (entry.cache_coerced && entry.cache_requested) {
            bits.push(`(asked ${entry.cache_requested})`);
        }
    }
    if (entry.flash_attn_applied) bits.push(`FA ${entry.flash_attn_applied}`);
    if (entry.n_ctx_applied) {
        // Format both numbers the same way; "ctx 262,144 / 8192 per seq" reads
        // like a typo next to the separated total.
        const seq = entry.n_ctx_seq && entry.n_ctx_seq !== entry.n_ctx_applied
            ? ` / ${entry.n_ctx_seq.toLocaleString()} per seq` : "";
        bits.push(`ctx ${entry.n_ctx_applied.toLocaleString()}${seq}`);
    }
    if (entry.weights_gib != null) bits.push(`${entry.weights_gib} GiB weights`);
    if (entry.kv_gib_estimate != null) bits.push(`~${entry.kv_gib_estimate} GiB KV`);
    if (entry.spec_type && entry.spec_type !== "none") bits.push(`spec ${entry.spec_type}`);
    if (entry.nextn_layers) bits.push(`${entry.nextn_layers} nextn`);
    if (entry.cpu_moe) bits.push(`MoE on CPU x${entry.n_cpu_moe || "all"}`);
    if (entry.load_seconds != null) bits.push(`loaded in ${entry.load_seconds}s`);
    const head = backendLine ? `backend ${backendLine}\n` : "";
    return head + bits.join(" | ");
}

function styleBtn(label, title) {
    const btn = document.createElement("button");
    btn.textContent = label;
    btn.title = title;
    Object.assign(btn.style, {
        padding: "3px 7px", borderRadius: "4px", border: "1px solid #555",
        background: "#2a2a2a", color: "#ddd", cursor: "pointer", fontSize: "11px",
    });
    return btn;
}

function addControls(node) {
    if (node.llamaCppUx) return;
    node.llamaCppUx = true;

    const widgetsByName = {};
    for (const w of node.widgets || []) widgetsByName[w.name] = w;

    const bar = document.createElement("div");
    Object.assign(bar.style, {
        display: "flex", gap: "3px", flexWrap: "wrap", justifyContent: "center",
        padding: "3px",
    });
    const status = document.createElement("div");
    Object.assign(status.style, {
        whiteSpace: "pre-wrap", fontSize: "10px", lineHeight: "1.35",
        color: "#9ecbff", padding: "3px 6px", textAlign: "left",
    });

    const wrap = document.createElement("div");
    wrap.appendChild(bar);
    wrap.appendChild(status);

    const hiddenGroups = new Set(COLLAPSED_BY_DEFAULT);

    const apply = () => {
        const perWidget = visibilityFor(widgetsByName);
        for (const [name, group] of Object.entries(SECTIONS)) {
            const groupShown = !hiddenGroups.has(name);
            for (const widgetName of group.widgets) {
                const widget = widgetsByName[widgetName];
                if (!widget) continue;
                setWidgetVisible(widget, groupShown && perWidget[widgetName] !== false);
            }
        }
        const warn = kvWarning(widgetsByName);
        // The warning line is re-derived every pass, so strip the previous one
        // before appending the current one instead of stacking duplicates.
        const stripped = status.textContent.replace(/\n⚠ [^\n]*/g, "");
        status.textContent = warn ? `${stripped}\n⚠ ${warn}` : stripped;

        node.setDirtyCanvas(true, true);
        node.setSize(node.computeSize());
    };

    for (const [name, group] of Object.entries(SECTIONS)) {
        const btn = styleBtn(group.label, `${group.hint} - click to show/hide`);
        const paint = () => {
            const on = !hiddenGroups.has(name);
            btn.style.background = on ? "#3a7bd5" : "#2a2a2a";
            btn.style.color = on ? "#fff" : "#888";
        };
        btn.onclick = () => {
            hiddenGroups.has(name) ? hiddenGroups.delete(name) : hiddenGroups.add(name);
            paint();
            apply();
        };
        group.paint = paint;
        paint();
        bar.appendChild(btn);
    }

    const refresh = async () => {
        try {
            const payload = await fetchStatus();
            const entry = payload.status?.[String(node.id)];
            const body = entry ? formatStatus(entry, payload.backend) : "";
            // Preserve the warning line that apply() appends.
            status.textContent = body;
            status.style.color = entry?.cache_coerced ? "#ffd479" : "#9ecbff";
            apply();
        } catch (err) {
            status.textContent = `applied settings unavailable: ${err}`;
            status.style.color = "#e08080";
        }
    };

    node.addDOMWidget("llama_cpp_loader_ux", "btn_bar", wrap, {
        serialize: false,
        hideOnZoom: false,
    });

    // Any widget change can flip what should be visible.
    for (const w of node.widgets || []) {
        const original = w.callback;
        w.callback = (...args) => {
            const result = original?.apply(w, args);
            if (["spec_type", "load_mtp", "cpu_moe", "flash_attn", "cache_type_k", "cache_type_v"]
                .includes(w.name)) {
                apply();
            }
            return result;
        };
    }

    Object.values(SECTIONS).forEach((g) => g.paint?.());
    apply();
    refresh();
    node.llamaCppRefresh = refresh;
}

app.registerExtension({
    name: "llama_cpp_vlm.loader_ux",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== LOADER) return;
        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onCreated?.apply(this, arguments);
            try {
                addControls(this);
            } catch (err) {
                console.warn("[llama-cpp_vlm] loader UI failed:", err);
            }
            return result;
        };
    },

    // A model load is what changes the applied numbers, so listen for it.
    setup() {
        api.addEventListener("executed", ({ detail }) => {
            if (!detail?.node_id) return;
            const node = app.graph?._nodes_by_id?.[detail.node_id];
            if (node?.comfyClass === LOADER && node.llamaCppRefresh) node.llamaCppRefresh();
        });
    },
});
