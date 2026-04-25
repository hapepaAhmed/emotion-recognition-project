const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const result = document.getElementById("result");

const model = localStorage.getItem("model");

// show selected model (optional UI improvement)
console.log("Using model:", model);

// ---------------- START CAMERA ----------------
async function startCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: true,
            audio: false
        });

        video.srcObject = stream;

        // IMPORTANT for some browsers
        await video.play();

        console.log("Camera started successfully");

        // start prediction loop AFTER camera is ready
        setInterval(sendFrame, 1000);

    } catch (error) {
        console.error("Camera access error:", error);
        alert("Camera blocked or not allowed. Please enable permission.");
    }
}



// ---------------- SEND FRAME ----------------
async function sendFrame() {

    const ctx = canvas.getContext("2d");

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    ctx.drawImage(video, 0, 0);

    canvas.toBlob(async (blob) => {

        const formData = new FormData();
        formData.append("image", blob);

        const response = await fetch("http://127.0.0.1:5000/predict", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (data.emotion) {
            result.innerText = 
                `Emotion: ${data.emotion} | Model: ${data.model}`;
        }

    }, "image/jpeg");
}

// start immediately
window.onload = () => {
    startCamera();
};