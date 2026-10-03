const PHOTON_URL = "https://photon.komoot.io/api/";
const AMSTERDAM_BBOX = "4.72,52.28,5.04,52.43";
const MIN_CHARS = 3;
const DEBOUNCE_MS = 300;

const searchInput = document.getElementById("location-search");
const suggestionsBox = document.getElementById("suggestions");
const latInput = document.getElementById("lat");
const lonInput = document.getElementById("lon");

let debounceTimer;

function clearSuggestions() {
    suggestionsBox.textContent = "";
}

function isAmsterdam(props) {
    return props.city && props.city.toLowerCase() === "amsterdam";
}

function buildLabel(props) {
    let street = props.name;
    if (props.street) {
        street = props.street;
        if (props.housenumber) {
            street += " " + props.housenumber;
        }
    }
    return [street, props.postcode, props.city].filter(Boolean).join(", ");
}

function showSuggestions(features) {
    clearSuggestions();

    const results = features.filter(function (feature) {
        return isAmsterdam(feature.properties);
    });

    if (results.length === 0) {
        const message = document.createElement("div");
        message.textContent = "No Amsterdam addresses found";
        suggestionsBox.appendChild(message);
        return;
    }

    results.forEach(function (feature) {
        const [lon, lat] = feature.geometry.coordinates;
        const label = buildLabel(feature.properties);

        const item = document.createElement("div");
        item.textContent = label;
        item.className = "suggestion-item";
        item.style.cursor = "pointer";

        item.addEventListener("click", function () {
            searchInput.value = label;
            latInput.value = lat;
            lonInput.value = lon;
            console.log("Selected:", label, lat, lon);
            clearSuggestions();
        });

        suggestionsBox.appendChild(item);
    });
}

async function fetchSuggestions(query) {
    const params = new URLSearchParams({
        q: query,
        limit: 10,
        bbox: AMSTERDAM_BBOX,
    });

    try {
        const response = await fetch(PHOTON_URL + "?" + params.toString());
        if (!response.ok) {
            throw new Error("Photon returned status " + response.status);
        }
        const data = await response.json();

        if (searchInput.value.trim() !== query) {
            return;
        }

        showSuggestions(data.features);
    } catch (error) {
        console.error("Address search failed:", error);
        clearSuggestions();
    }
}

searchInput.addEventListener("input", function () {
    latInput.value = "";
    lonInput.value = "";

    clearTimeout(debounceTimer);

    const query = searchInput.value.trim();
    if (query.length < MIN_CHARS) {
        clearSuggestions();
        return;
    }

    debounceTimer = setTimeout(function () {
        fetchSuggestions(query);
    }, DEBOUNCE_MS);
});