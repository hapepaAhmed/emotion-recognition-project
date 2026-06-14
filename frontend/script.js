const video      = document.getElementById("video");
const canvas     = document.getElementById("canvas");
const snapshot   = document.getElementById("snapshot");
const result     = document.getElementById("result");
const captureBtn = document.getElementById("capture-btn");
const restartBtn = document.getElementById("restart-btn");
const overlay    = document.getElementById("predicting-overlay");
const modelLabel = document.getElementById("model-label");

const modelName  = localStorage.getItem("model") || "unknown";

// Show which model is active
const modelDisplayNames = {
    cnn: "Custom CNN",
    mobilenet: "MobileNet V2",
    efficientnet: "EfficientNet B0"
};
modelLabel.textContent = `Model: ${modelDisplayNames[modelName] || modelName}`;

let currentStream = null;

// ---------------- START CAMERA ----------------
async function startCamera() {
    // Reset UI to "live" state
    result.style.display    = "none";
    restartBtn.style.display = "none";
    snapshot.style.display  = "none";
    video.style.display     = "block";
    captureBtn.style.display = "none";  // show only once camera is ready

    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: "user" },
            audio: false
        });

        currentStream = stream;
        video.srcObject = stream;
        await video.play();

        // Show capture button once camera is live
        captureBtn.style.display = "inline-flex";

    } catch (error) {
        console.error("Camera access error:", error);
        modelLabel.textContent = "⚠ Camera blocked. Please enable permission.";
    }
}

// ---------------- STOP CAMERA ----------------
function stopCamera() {
    if (currentStream) {
        currentStream.getTracks().forEach(track => track.stop());
        currentStream = null;
    }
    video.srcObject = null;
}

// ---------------- CAPTURE & PREDICT ----------------
async function captureAndPredict() {
    // 1. Capture current frame to canvas
    canvas.width  = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0);

    // 2. Show the frozen snapshot, hide live video
    snapshot.src = canvas.toDataURL("image/jpeg");
    snapshot.style.display = "block";
    video.style.display    = "none";

    // 3. Stop the camera stream
    stopCamera();

    // 4. Hide capture btn, show spinner overlay
    captureBtn.style.display = "none";
    overlay.style.display    = "flex";

    // 5. Convert canvas to blob and send to backend
    canvas.toBlob(async (blob) => {
        const formData = new FormData();
        formData.append("image", blob);

        try {
            const response = await fetch("http://127.0.0.1:5000/predict", {
                method: "POST",
                body: formData
            });

            const data = await response.json();

            overlay.style.display = "none";
            result.style.display  = "block";

            if (data.emotion) {
                const emoji = { Happy: "😄", Neutral: "😐", Sad: "😢" };
                result.innerHTML =
                    `<span class="emotion-emoji">${emoji[data.emotion] || "🤔"}</span>
                     <span class="emotion-text">${data.emotion}</span>
                     <span class="model-tag">${modelDisplayNames[data.model] || data.model}</span>`;
            } else if (data.error) {
                result.innerHTML = `<span class="emotion-text" style="color:#f87171;">⚠ ${data.error}</span>`;
            }

        } catch (err) {
            overlay.style.display = "none";
            result.style.display  = "block";
            result.innerHTML = `<span class="emotion-text" style="color:#f87171;">⚠ Backend not reachable</span>`;
            console.error("Prediction error:", err);
        }

        // 6. Show restart button
        restartBtn.style.display = "inline-flex";

    }, "image/jpeg", 0.92);
}

// ---------------- RESTART ----------------
function restartCamera() {
    startCamera();
}

// ---------------- INIT ----------------
window.onload = () => {
    startCamera();
};