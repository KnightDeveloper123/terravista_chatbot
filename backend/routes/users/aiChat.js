const express = require('express');
const router = express.Router();
const { middleware } = require('../../middleware/middleware')
const connection = require('../../database/db');

router.get("/get-info", async (req, res) => {
    try {
        const { query } = req.query;
        if (!query) {
            return res.status(400).json({ error: "Query parameter is required" });
        }
        const pythonApiUrl = process.env.PYTHON_BOT_URL;

        const response = await fetch(`${pythonApiUrl}/stream_info?query=${query}`, {
            headers: { "Content-Type": "application/json" },
            method: "GET"
        });

        if (!response.ok) {
            return res.status(response.status).json({ error: "Failed to fetch from Python bot" });
        }
        const data = await response.json();

        console.log(data);

        return res.json({ success: "success", answer: data?.answer || "No answer received" });


    } catch (error) {
        console.error("Error in /get-info:", error);
        res.status(500).json({ error: "Internal Server Error" });
    }
});


router.post("/get-info", async (req, res) => {
    try {
        const { title_id } = req.query;
        console.log(title_id);

        if (!title_id) {
            return res.status(400).json({ error: "title id field is required" });
        }

        const pythonApiUrl = process.env.PYTHON_BOT_URL;

        // Send POST request to Python API
        const response = await fetch(`${pythonApiUrl}/stream_info`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ title_id })
        });

        if (!response.ok) {
            return res.status(response.status).json({ error: "Failed to fetch from Python bot" });
        }

        const data = await response.json();
        console.log("Response from Python bot:", data);

        return res.json({
            success: "success",
            answer: data?.answer || "No answer received"
        });
    } catch (error) {
        console.error("Error in /get-info:", error);
        res.status(500).json({ error: "Internal Server Error" });
    }
});


module.exports = router;