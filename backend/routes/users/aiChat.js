const express = require('express');
const router = express.Router();

router.get("/get-info", async (req, res) => {
    try {
        const { title_id, query } = req.query;

        if (!query) {
            return res.status(400).json({ error: "Query parameter is required" });
        }

        const pythonApiUrl = process.env.PYTHON_BOT_URL;

        const response = await fetch(`${pythonApiUrl}/stream_info`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ title_id: title_id ?? 0, query: query })
        });

        if (!response.ok) {
            return res.status(response.status).json({ error: "Failed to fetch from Python bot" });
        }

        console.log('here');


        res.setHeader("Content-Type", "text/event-stream");
        res.setHeader("Cache-Control", "no-cache");
        res.setHeader("Connection", "keep-alive");

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            if (!value) continue;

            buffer += decoder.decode(value, { stream: true });

            // split by words or sentences, depending on how you want to stream
            let boundary = buffer.indexOf(" "); // send word by word
            while (boundary !== -1) {
                const chunk = buffer.slice(0, boundary + 1);
                buffer = buffer.slice(boundary + 1);

                if (chunk.trim()) {
                    res.write(`data: ${chunk}\n\n`);
                }

                boundary = buffer.indexOf(" ");
            }
        }

        // flush remaining buffer
        if (buffer.trim()) res.write(`data: ${buffer}\n\n`);

        // signal end of stream
        res.write("data: end\n\n");
        res.end();


        // const data = await response.json();
        // console.log("Response from Python bot:", data);

        // return res.json({ success: "success", answer: data?.answer || "No answer received" });
    } catch (error) {
        console.error("Error in /get-info:", error);
        res.status(500).json({ error: "Internal Server Error" });
    }
});


module.exports = router;