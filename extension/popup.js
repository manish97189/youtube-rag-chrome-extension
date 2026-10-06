let currentVideoId = null;

const statusEl = document.getElementById("video-status");
const chatBox = document.getElementById("chat-box");
const inputEl = document.getElementById("question-input");
const askBtn = document.getElementById("ask-btn");

function extractVideoId(url) {
  try {
    const parsed = new URL(url);
    if (parsed.hostname.includes("youtube.com")) {
      return parsed.searchParams.get("v");
    }
    if (parsed.hostname.includes("youtu.be")) {
      return parsed.pathname.slice(1);
    }
  } catch (e) {
    return null;
  }
  return null;
}

function appendMessage(text, sender) {
  const div = document.createElement("div");
  div.className = `msg ${sender}`;
  div.textContent = text;
  chatBox.appendChild(div);
  chatBox.scrollTop = chatBox.scrollHeight;
  return div;
}

// 1. Grab the YouTube Video ID from the active tab
chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
  const url = tabs[0]?.url || "";
  currentVideoId = extractVideoId(url);

  if (currentVideoId) {
    statusEl.textContent = `Connected to Video ID: ${currentVideoId}`;
    inputEl.disabled = false;
    askBtn.disabled = false;
  } else {
    statusEl.textContent = "Please open a YouTube video first.";
  }
});

// 2. Send question + video_id to FastAPI
async function handleAsk() {
  const question = inputEl.value.trim();
  if (!question || !currentVideoId) return;

  appendMessage(question, "user");
  inputEl.value = "";
  askBtn.disabled = true;

  const loadingMsg = appendMessage("Indexing & thinking...", "bot");

  try {
    const response = await fetch("http://127.0.0.1:8000/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ video_id: currentVideoId, question })
    });

    const data = await response.json();
    if (!response.ok) {
      loadingMsg.textContent = `Error: ${data.detail || "Something went wrong"}`;
    } else {
      loadingMsg.textContent = data.answer;
    }
  } catch (err) {
    loadingMsg.textContent = "Cannot connect to FastAPI server on port 8000.";
  } finally {
    askBtn.disabled = false;
  }
}

askBtn.addEventListener("click", handleAsk);
inputEl.addEventListener("keypress", (e) => {
  if (e.key === "Enter") handleAsk();
});