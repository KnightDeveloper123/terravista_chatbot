const connection = require("../database/db");
const executeQuery = require("../utils/executeQuery");

async function getAiReply(titleId, userId, name, query) {
    try {
        if (!titleId || !userId || !query) {
            return { error: 'titleId, userId and query are required' }
        }
        const pythonApiUrl = process.env.PYTHON_BOT_URL;

        const response = await fetch(pythonApiUrl + "/get_info", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                title_id: titleId,
                query: query,
                user_id: userId,
                name: name
            })
        });

        if (!response.ok) {
            throw new Error("Failed to fetch from Python bot");
        }

        const data = await response.json();

        console.log(data);

        if (data.success) {
            return { ...data }
        } else {
            return { error: "Sorry, I'm having trouble responding right now." }
        }
    } catch (error) {
        console.log("AI Service Error:", error);
        return { error: "Sorry, I'm having trouble responding right now." };
    }
}


/**
    let isStreaming = true;
        let fullText = "";
        let buffer = "";

        console.log("Streaming");


        while (isStreaming) {
            const { value, done } = await reader.read();
            if (done) {
                isStreaming = false;
                break;
            };

            if (!value) continue;

            buffer += decoder.decode(value, { stream: true });
            // console.log(buffer);
            let boundary = buffer.indexOf(" ");
            while (boundary !== -1) {
                const chunk = buffer.slice(0, boundary + 1);
                buffer = buffer.slice(boundary + 1);

                if (chunk.trim()) {
                    fullText += chunk;
                }

                boundary = buffer.indexOf(" ");
            }
        }

        if (buffer.trim()) {
            fullText += buffer;
        }
        // console.log(fullText);

        return fullText;

 **/

async function getOrCreateWhatsAppTitle(phone) {
    const users = await executeQuery("SELECT id, name FROM user WHERE phone_number = ?", [phone]);

    let userId;
    let titleId;

    if (users.length === 0) {
        const addUser = await executeQuery("INSERT INTO user (phone_number) VALUES (?)", [phone]);
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

    return { userId, titleId, name: users[0]?.name ?? null };
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
