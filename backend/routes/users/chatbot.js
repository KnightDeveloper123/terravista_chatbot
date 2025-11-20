const express = require('express');
const router = express.Router();
const { middleware } = require('../../middleware/middleware')
const connection = require('../../database/db');
const executeQuery = require('../../utils/executeQuery');

router.post("/addChat", middleware, (req, res) => {
    const { message, sender, title_id } = req.body;
    console.log(message, "addChat", title_id);

    const query = 'INSERT INTO chats (title_id,message,sender) VALUES (?,?,?)'
    const values = [title_id, message, sender]

    connection.query(query, values, (err, result) => {
        if (err) {
            console.error("Error inserting chat:", err);
            return res.status(500).json({ error: "Database error" });
        }
        res.status(201).json({ message: "Chat Added", chat_id: result.insertId });
    });
});

router.post("/newChat", middleware, async (req, res) => {

    const { chats, user_id } = req.body;
    console.log(chats, "chats");


    const message = `New Chat ${new Date().toLocaleDateString("en-GB", { day: 'numeric', month: 'long', year: 'numeric' })}`;
    const query = 'INSERT INTO chat_titles (title, user_id) VALUES (?,?)'
    const values = [message, user_id]

    connection.query(query, values, (err, result) => {
        if (err) {
            console.error("Error inserting chat:", err);
            return res.status(500).json({ error: "Database error" });
        }
        const title_id = result.insertId;
        const query2 = 'INSERT INTO chats (message, sender, title_id) VALUES (?, ?, ?)';


        const values2 = [
            chats?.message, chats?.sender, title_id
        ];

        connection.query(query2, values2, (err, data) => {
            if (err) {
                console.error("Error inserting chat messages:", err);
                return res.status(500).json({ error: "Database error" });
            }
            res.status(201).json({ message: "Chat created successfully", chat_id: title_id, data });
        });


    });

});

router.delete("/deleteChatTitle", middleware, (req, res) => {
    const { title_id } = req.body;

    const sql = "update chat_titles set status=1 WHERE id=?";
    const value = [title_id];

    connection.query(sql, value, (err, data) => {
        if (err) {
            console.error("Error deleting chat:", err);
            return res.status(500).json({ error: "Database error" });
        }
        return res.status(201).json({ message: "Chat Title deleted successfully", data });
    })
})

router.get("/getChatTitle", (req, res) => {
    const { user_id } = req.query;

    const query = 'select * from chat_titles where user_id=? and status=0'
    const value = [user_id];
    connection.query(query, value, (err, data) => {
        if (err) {
            console.error("Error fetching sidebar data:", err);
            return res.status(400).json({ error: "Database query failed" });
        }
        return res.json({ success: "success", data });
    });
})

router.get("/getAllChats", (req, res) => {
    const { title_id } = req.query;
    connection.query(`select * from chats where title_id=? `, [title_id], (err, data) => {
        if (err) {
            console.error("Error fetching sidebar data:", err);
            return res.status(500).json({ error: "Database query failed" });
        }
        res.json({ success: "success", data });
    });

})

router.post("/reaction", middleware, async (req, res) => {
    try {
        const { chat_id, reaction } = req.body;

        if (!chat_id || (reaction !== 0 && reaction !== 1)) {
            return res.status(400).json({ error: "Invalid input. Reaction must be 0 or 1." });
        }

        const [chatDetail] = await executeQuery("SELECT * FROM chats WHERE id = ?", [chat_id]);

        if (!chatDetail) {
            return res.status(404).json({ error: "Chat message not found." });
        }

        // Check if already reacted
        if (chatDetail.reaction !== null) {
            return res.status(400).json({ error: "User has already reacted to this chat." });
        }

        await executeQuery("UPDATE chats SET reaction = ? WHERE id = ?", [reaction, chat_id]);
        return res.json({ success: true, message: "Reaction saved successfully." });
    } catch (err) {
        console.error("Error in /reaction:", err);
        res.status(500).json({ error: "Internal Server Error" });
    }
});

router.post("/feedback", middleware, (req, res) => {

    const { chat_id, user_id, feedback } = req.body;
    console.log(chat_id, user_id, feedback);


    connection.query(`INSERT INTO chat_feedback (chat_id,user_id,feedback) VALUES (?,?,?)`, [chat_id, user_id, feedback],
        (err, data) => {
            if (err) {
                console.error("Error fetching sidebar data:", err);
                return res.status(500).json({ error: "Database query failed" });
            }
            res.json({ success: "success", data });
        });

})

router.get("/getChatFeedback", middleware, (req, res) => {
    connection.query(`select * from chat_feedback where feedback=0 || feedback=1`, (err, data) => {
        if (err) {
            console.error("Error fetching sidebar data:", err);
            return res.status(500).json({ error: "Database query failed" });
        }
        res.json({ success: "success", data });
    });
})

module.exports = router;