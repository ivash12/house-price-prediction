const PHOTON_URL = "https://photon.komoot.io/api/";
const MIN_CHARS = 3;
const DEBOUNCE_MS = 300;

const searchInput = document.getElementById("location-search");
const suggestionsBox = document.getElementById("suggestions");
const latInput = document.getElementById("lat");
const lonInput = document.getElementById("lon");

const CITIES = {
    "amsterdam": { name: "Amsterdam", bbox: "4.72,52.28,5.04,52.43", cityFilter: "amsterdam" },
    "new-york": { name: "New York", bbox: "-74.27,40.49,-73.68,40.92", cityFilter: null },
};
const city = CITIES[searchInput.dataset.city];

let debounceTimer;

function clearSuggestions() {
    suggestionsBox.textContent = "";
}

function isInCity(props) {
    if (!city.cityFilter) return true;
    return props.city && props.city.toLowerCase() === city.cityFilter;
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
        return isInCity(feature.properties);
    });

    if (results.length === 0) {
        const message = document.createElement("div");
        message.textContent = "No " + city.name + " addresses found";
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
        bbox: city.bbox,
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
