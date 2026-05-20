const startBtn = document.getElementById('start-btn');
const computeSelect = document.getElementById('compute-level');
const logs = document.getElementById('logs');
let worker = null;

function log(msg, type = "info") {
    const p = document.createElement('div');
    p.textContent = `> ${msg}`;
    p.style.color = type === "error" ? "#ef4444" : "#10b981";
    logs.appendChild(p);
    logs.scrollTop = logs.scrollHeight;
}

startBtn.addEventListener('click', () => {
    if (worker) {
        log("Stopping volunteer compute...", "error");
        worker.terminate();
        worker = null;
        startBtn.textContent = "Start Volunteering";
        startBtn.style.background = "var(--accent)";
        computeSelect.disabled = false;
        return;
    }

    const threads = parseInt(computeSelect.value, 10);
    log(`Initializing Web Worker with ${threads} threads...`);
    
    worker = new Worker('fl_worker.js');
    
    worker.onmessage = (e) => {
        const { type, msg } = e.data;
        if (type === "STATUS") {
            log(msg);
        } else if (type === "RESULT") {
            log(msg);
        } else if (type === "ERROR") {
            log(msg, "error");
        }
    };
    
    worker.postMessage({
        type: "INIT",
        payload: { threads }
    });
    
    startBtn.textContent = "Stop Volunteering";
    startBtn.style.background = "#ef4444";
    computeSelect.disabled = true;
});
