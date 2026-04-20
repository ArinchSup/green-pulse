const form = document.getElementById("stock-form");
const symbolInput = document.getElementById("symbol");
const status = document.getElementById("status");
const canvas = document.getElementById("graph");
const ctx = canvas.getContext("2d");
const authMessage = document.getElementById("auth-message");
const loginLink = document.getElementById("login-link");
const signupLink = document.getElementById("signup-link");
const logoutLink = document.getElementById("logout-link");
const exitButton = document.getElementById("exit-button");
const footer = document.getElementById("footer");

function performLogout(event) {
	if (event && typeof event.preventDefault === "function") {
		event.preventDefault();
	}
	localStorage.removeItem("accessToken");
	localStorage.removeItem("userEmail");
	localStorage.removeItem("userId");
	sessionStorage.removeItem("accessToken");
	sessionStorage.removeItem("userEmail");
	sessionStorage.removeItem("userId");

	const authNav = document.getElementById("auth-nav");
	const userInfo = document.getElementById("user-info");
	const userEmail = document.getElementById("user-email");
	const favoritesPanel = document.getElementById("favorites-panel");
	const favoritesList = document.getElementById("favorites-list");

	if (authNav) {
		authNav.style.display = "flex";
	}
	if (userInfo) {
		userInfo.style.display = "none";
	}
	if (userEmail) {
		userEmail.textContent = "";
	}
	if (form) {
		form.style.display = "none";
	}
	if (authMessage) {
		authMessage.style.display = "block";
	}
	if (status) {
		status.textContent = "Not authenticated.";
	}
	if (footer) {
		footer.style.display = "none";
	}
	if (favoritesPanel) {
		favoritesPanel.classList.remove("show");
	}
	if (favoritesList) {
		favoritesList.innerHTML = '<div class="favorites-empty">No favorites yet</div>';
	}

	const cleanUrl = `${window.location.origin}${window.location.pathname}`;
	window.location.replace(cleanUrl);
}

window.performLogout = performLogout;

loginLink.addEventListener("click", (e) => {
	e.preventDefault();
	window.location.href = "http://localhost:8080/signin";
});

signupLink.addEventListener("click", (e) => {
	e.preventDefault();
	const existingEmail = localStorage.getItem("userEmail");
	if (existingEmail) {
		alert(`Account already exists for ${existingEmail}. You are already logged in!`);
		return;
	}
	window.location.href = "http://localhost:8080/signup";
});

if (logoutLink) {
	logoutLink.addEventListener("click", (e) => {
		performLogout(e);
	});
}

document.addEventListener("click", (e) => {
	const target = e.target;
	if (!(target instanceof Element)) {
		return;
	}
	if (target.id === "logout-link") {
		performLogout(e);
	}
});

exitButton.addEventListener("click", () => {
	if (confirm("Are you sure you want to shutdown the server?")) {
		window.location.href = "http://localhost:8080/exit";
	}
});

function checkAuth() {
	const params = new URLSearchParams(window.location.search);
	const token = params.get("token");
	const email = params.get("email");
	const userId = params.get("user_id");
	const signupStatus = params.get("signup_status");

	if (signupStatus === "complete") {
		alert(`Signup complete for ${email || "this account"}. Please click sign in.`);
		window.history.replaceState({}, document.title, window.location.pathname);
	} else if (signupStatus === "exists") {
		alert(`Account already exists for ${email || "this email"}. Please click sign in.`);
		window.history.replaceState({}, document.title, window.location.pathname);
	}

	if (token) {
		localStorage.setItem("accessToken", token);
		if (email) {
			localStorage.setItem("userEmail", email);
		}
		if (userId) {
			localStorage.setItem("userId", userId);
		}
		window.history.replaceState({}, document.title, window.location.pathname);
	}

	const storedToken = localStorage.getItem("accessToken");
	const storedEmail = localStorage.getItem("userEmail");
	const storedUserId = localStorage.getItem("userId");

	const authNav = document.getElementById("auth-nav");
	const userInfo = document.getElementById("user-info");
	const userEmail = document.getElementById("user-email");
	const favoritesPanel = document.getElementById("favorites-panel");

	if (!storedToken) {
		form.style.display = "none";
		authMessage.style.display = "block";
		status.textContent = "Not authenticated.";
		authNav.style.display = "flex";
		userInfo.style.display = "none";
		footer.style.display = "none";
		favoritesPanel.classList.remove("show");
		return false;
	}

	authNav.style.display = "none";
	userInfo.style.display = "flex";
	footer.style.display = "block";
	favoritesPanel.classList.add("show");
	if (storedEmail) {
		userEmail.textContent = storedEmail;
	}
	if (storedUserId) {
		loadFavorites(storedUserId);
	}
	return true;
}

if (!checkAuth()) {
	ctx.clearRect(0, 0, canvas.width, canvas.height);
}

let currentSymbol = null;
const favoriteButton = document.getElementById("favorite-button");
const loginPromptLink = document.getElementById("login-prompt-link");

loginPromptLink.addEventListener("click", (e) => {
	e.preventDefault();
	window.location.href = "http://localhost:8080/signin";
});

async function loadFavorites(userId) {
	try {
		const response = await fetch(`http://localhost:8080/favorites?user_id=${userId}`);
		if (!response.ok) return;
		const data = await response.json();
		displayFavorites(data || []);
	} catch (error) {
		console.error("Failed to load favorites:", error);
	}
}

function displayFavorites(favorites) {
	const list = document.getElementById("favorites-list");
	if (favorites.length === 0) {
		list.innerHTML = '<div class="favorites-empty">No favorites yet</div>';
		return;
	}
	list.innerHTML = favorites.map((fav) => `
		<div class="favorite-item">
			<span style="cursor: pointer; flex: 1;" onclick="loadSymbol('${fav.symbol}')">${fav.symbol}</span>
			<button type="button" onclick="removeFavorite('${fav.symbol}')">×</button>
		</div>
	`).join("");
}

function loadSymbol(symbol) {
	symbolInput.value = symbol;
	form.dispatchEvent(new Event("submit"));
}

async function addToFavorites(symbol) {
	const userId = localStorage.getItem("userId");
	if (!userId) return;
	try {
		const response = await fetch(`http://localhost:8080/favorites?user_id=${userId}&symbol=${symbol}`, {
			method: "POST",
		});
		if (response.ok) {
			loadFavorites(userId);
		}
	} catch (error) {
		console.error("Failed to add favorite:", error);
	}
}

async function removeFavorite(symbol) {
	const userId = localStorage.getItem("userId");
	if (!userId) return;
	try {
		const response = await fetch(`http://localhost:8080/favorites?user_id=${userId}&symbol=${symbol}`, {
			method: "DELETE",
		});
		if (response.ok) {
			loadFavorites(userId);
		}
	} catch (error) {
		console.error("Failed to remove favorite:", error);
	}
}

favoriteButton.addEventListener("click", (e) => {
	e.preventDefault();
	if (currentSymbol) {
		addToFavorites(currentSymbol);
	}
});

function formatDate(value) {
	const date = new Date(value);
	return date.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
}

function drawGraph(records, symbol) {
	const width = canvas.width;
	const height = canvas.height;
	const padding = { top: 28, right: 34, bottom: 48, left: 64 };
	const chartWidth = width - padding.left - padding.right;
	const chartHeight = height - padding.top - padding.bottom;

	ctx.clearRect(0, 0, width, height);
	ctx.fillStyle = "#0f172a";
	ctx.fillRect(0, 0, width, height);

	if (!records.length) {
		ctx.fillStyle = "#e2e8f0";
		ctx.font = "18px system-ui, sans-serif";
		ctx.fillText("No chart data available.", padding.left, padding.top + 20);
		return;
	}

	const sorted = [...records].sort((left, right) => new Date(left.date).getTime() - new Date(right.date).getTime());
	const closes = sorted.map((record) => record.close);
	const minClose = Math.min(...closes);
	const maxClose = Math.max(...closes);
	const range = maxClose - minClose || 1;

	const xFor = (index) => padding.left + (sorted.length === 1 ? chartWidth / 2 : (index / (sorted.length - 1)) * chartWidth);
	const yFor = (value) => padding.top + chartHeight - ((value - minClose) / range) * chartHeight;

	ctx.strokeStyle = "rgba(148, 163, 184, 0.22)";
	ctx.lineWidth = 1;
	ctx.beginPath();
	ctx.moveTo(padding.left, padding.top);
	ctx.lineTo(padding.left, padding.top + chartHeight);
	ctx.lineTo(padding.left + chartWidth, padding.top + chartHeight);
	ctx.stroke();

	const stepCount = 4;
	ctx.font = "12px system-ui, sans-serif";
	ctx.fillStyle = "#cbd5e1";
	for (let index = 0; index <= stepCount; index += 1) {
		const value = minClose + (range * index) / stepCount;
		const y = yFor(value);
		ctx.strokeStyle = "rgba(148, 163, 184, 0.16)";
		ctx.beginPath();
		ctx.moveTo(padding.left, y);
		ctx.lineTo(padding.left + chartWidth, y);
		ctx.stroke();
		ctx.fillText(value.toFixed(2), 12, y + 4);
	}

	ctx.strokeStyle = "#38bdf8";
	ctx.lineWidth = 2.5;
	ctx.beginPath();
	sorted.forEach((record, index) => {
		const x = xFor(index);
		const y = yFor(record.close);
		if (index === 0) {
			ctx.moveTo(x, y);
		} else {
			ctx.lineTo(x, y);
		}
	});
	ctx.stroke();

	ctx.fillStyle = "rgba(56, 189, 248, 0.18)";
	ctx.lineTo(padding.left + chartWidth, padding.top + chartHeight);
	ctx.lineTo(padding.left, padding.top + chartHeight);
	ctx.closePath();
	ctx.fill();

	ctx.strokeStyle = "#38bdf8";
	ctx.fillStyle = "#e2e8f0";
	ctx.font = "600 16px system-ui, sans-serif";
	ctx.fillText(`${symbol.toUpperCase()} all-time close`, padding.left, 18);

	ctx.font = "12px system-ui, sans-serif";
	ctx.fillStyle = "#cbd5e1";
	ctx.fillText(formatDate(sorted[0].date), padding.left, height - 16);
	const lastLabel = formatDate(sorted[sorted.length - 1].date);
	const lastLabelWidth = ctx.measureText(lastLabel).width;
	ctx.fillText(lastLabel, width - padding.right - lastLabelWidth, height - 16);
	ctx.fillText(`min ${minClose.toFixed(2)} / max ${maxClose.toFixed(2)}`, padding.left, padding.top + 18);
}

form.addEventListener("submit", async (event) => {
	event.preventDefault();

	const symbol = symbolInput.value.trim();
	if (!symbol) {
		status.textContent = "Please enter a symbol.";
		return;
	}

	status.textContent = "Loading chart...";
	ctx.clearRect(0, 0, canvas.width, canvas.height);
	favoriteButton.style.display = "none";

	try {
		const response = await fetch(`http://localhost:8080/watchlist?symbol=${encodeURIComponent(symbol)}`);
		const text = await response.text();

		if (!response.ok) {
			status.textContent = text;
			return;
		}

		const payload = JSON.parse(text);
		currentSymbol = payload.symbol || symbol;
		drawGraph(payload.records || [], currentSymbol);
		status.textContent = `${currentSymbol} loaded with ${payload.count || 0} records.`;
		if (localStorage.getItem("userId")) {
			favoriteButton.style.display = "inline-block";
		}
	} catch (error) {
		status.textContent = "request failed";
	}
});
