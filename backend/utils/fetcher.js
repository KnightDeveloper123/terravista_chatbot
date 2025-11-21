async function fetcher(url, method, headers, body) {
    try {
        const options = {
            method: method || "GET",
            headers: headers || {}
        };

        if (body) {
            options.body = JSON.stringify(body);
            options.headers["Content-Type"] = "application/json";
        }

        const response = await fetch(url, options);

        const contentType = response.headers.get("content-type");

        if (!response.ok) {
            const text = await response.text();
            throw new Error("Fetch failed: " + text);
        }

        if (contentType && contentType.includes("application/json")) {
            return await response.json();
        } else {
            return await response.text();
        }

    } catch (error) {
        console.log("Fetcher Error:", error.message);
        throw error;
    }
}

module.exports = fetcher;
