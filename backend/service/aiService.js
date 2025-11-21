const connection = require("../database/db");
const executeQuery = require("../utils/executeQuery");

async function getAiReply(titleId, userId, query) {
    try {
        if (!titleId || !userId || !query) {
            return { error: 'titleId, userId and query are required' }
        }
        const pythonApiUrl = process.env.PYTHON_BOT_URL;

        const response = await fetch(pythonApiUrl + "/stream_info", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                title_id: titleId,
                query: query,
                user_id: userId
            })
        });

        if (!response.ok) {
            throw new Error("Failed to fetch from Python bot");
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        let isStreaming = true;
        let fullText = "";
        let buffer = "";

        while (isStreaming) {
            const { value, done } = await reader.read();
            if (done) {
                isStreaming = false;
                break;
            };
            if (!value) continue;

            buffer += decoder.decode(value, { stream: true });

            let boundary = buffer.indexOf(" ");
            while (boundary !== -1) {
                const chunk = buffer.slice(0, boundary + 1);
                buffer = buffer.slice(boundary + 1);

                if (chunk.trim()) {
                    fullText += chunk; // store chunk
                }

                boundary = buffer.indexOf(" ");
            }
        }

        if (buffer.trim()) {
            fullText += buffer;
        }

        return { success: true, data: fullText.trim() };

    } catch (error) {
        console.log("AI Service Error:", error.message);
        return "Sorry, I'm having trouble responding right now.";
    }
}

async function getOrCreateWhatsAppTitle(name, phone) {
    const users = await executeQuery("SELECT id FROM user WHERE phone_number = ?", [phone]);

    let userId;
    let titleId;

    if (users.length === 0) {
        const addUser = await executeQuery("INSERT INTO user (name, phone_number) VALUES (?, ?)", [name, phone]);
        if (!addUser.insertId) {
            throw new Error("Failed to insert new user");
        }
        userId = addUser.insertId;
    } else {
        userId = users[0].id;
    }

    const titles = await executeQuery("select * from chat_titles where user_id = ? AND status = 0 order by created_at LIMIT 1", [userId]);

    const titleName = `New Chat ${new Date().getDate()} ${["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"][new Date().getMonth()]} ${new Date().getFullYear()}`;

    if (titles.length === 0) {
        const addTitle = await executeQuery('INSERT INTO chat_titles (title, user_id) VALUES (?, ?)', [titleName, userId]);
        if (!addTitle.insertId) {
            throw new Error("Failed to insert new title");
        }
        titleId = addTitle.insertId;
    } else {
        titleId = titles[0].id;
    }

    return { userId, titleId };
}

async function saveChatMessage(titleId, message, sender) {
    try {
        if (!titleId || !message || !sender) {
            return { error: "titleId, message, and sender are required" };
        }

        const query = 'INSERT INTO chats (title_id,message,sender) VALUES (?,?,?)'
        const values = [titleId, message, sender]

        connection.query(query, values, (err, result) => {
            if (err) {
                console.error("Error inserting chat:", err);
                return { error: "Database error" };
            }
            return { success: "Chat Added" };
        });

    } catch (err) {
        console.log("Chat Save Error:", err.message);
        return {
            success: false,
            error: "Database error",
            data: null
        };
    }
}

module.exports = { getAiReply, getOrCreateWhatsAppTitle, saveChatMessage };
