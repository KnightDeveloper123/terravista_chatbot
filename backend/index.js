const express = require("express");
const cors = require("cors");
require('dotenv').config()
const path = require("path");
const { getOrCreateWhatsAppTitle, saveChatMessage, getAiReply } = require("./service/aiService");



const app = express();

const allowedOrigin = process.env.FRONTEND_URL; // your frontend

app.use(
    cors({
        origin: allowedOrigin,  // must be specific when credentials are included
        credentials: true,      // allow cookies/authorization headers
        methods: ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allowedHeaders: ["Content-Type", "Authorization"],
    })
);

app.use(express.json());

app.use("/documents", express.static(path.join(__dirname, "documents/")));
app.use("/brochures", express.static(path.join(__dirname, "brochures/")));
app.use("/profile", express.static(path.join(__dirname, "profile/")));

app.use(express.urlencoded({ extended: true }));
app.use(express.json());

app.use('/admin', require("./routes/admin/admin"))
app.use('/support', require("./routes/admin/support"))
app.use('/documents', require("./routes/admin/documents"))
app.use('/employee', require("./routes/admin/employee"))

app.use('/user', require("./routes/users/user"))
app.use('/community', require("./routes/users/community"))

app.use('/chatbot', require("./routes/users/chatbot"))
app.use('/ai', require('./routes/users/aiChat'));


app.use('/meetings', require('./routes/admin/meetings'));


app.get("/webhook", (req, res) => {
    const VERIFY_TOKEN = process.env.VERIFY_TOKEN;

    const mode = req.query["hub.mode"];
    const token = req.query["hub.verify_token"];
    const challenge = req.query["hub.challenge"];

    if (mode === "subscribe" && token === VERIFY_TOKEN) {
        console.log('here', challenge, mode);

        return res.status(200).send(challenge);
    }

    return res.sendStatus(403);
});

app.post("/webhook", async (req, res) => {
    try {
        const body = req.body;

        if (body.object === "whatsapp_business_account") {
            const entry = body.entry?.[0];
            const changes = entry?.changes?.[0];
            const messages = changes?.value?.messages;

            if (messages && messages.length > 0) {
                const msg = messages[0];
                const from = msg.from;
                const text = msg.text?.body;


                const userDetails = await getOrCreateWhatsAppTitle("WhatsApp User", from);
                if (userDetails.error) {
                    console.log("Error in getOrCreateWhatsAppTitle", userDetails.error)
                }
                const { userId, titleId } = userDetails;

                const saveMessage = await saveChatMessage(titleId, text, "user");
                if (saveMessage?.error) {
                    console.log("Error in saveChatMessage", saveMessage?.error)
                }

                const aiRes = await getAiReply(titleId, userId, text);


                if (aiRes.error) {
                    console.log("Error in getAiReply", aiRes.error)
                }
                const reply = aiRes?.data || "Sorry, I couldn't generate a reply.";

                await saveChatMessage(titleId, reply, "bot");

                console.log("aiRes:", aiRes);

                const wpRes = await fetch(`https://graph.facebook.com/v24.0/${process.env.PHONE_NUMBER_ID}/messages`, {
                    method: "POST",
                    headers: {
                        "Authorization": "Bearer " + process.env.META_ACCESS_TOKEN,
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify({
                        messaging_product: "whatsapp",
                        to: from,
                        type: "text",
                        text: { body: reply }
                    })
                });

                const wpJson = await wpRes.json();
                console.log("WHATSAPP API RESPONSE:", wpJson);
            }
        }

        res.sendStatus(200);
    } catch (error) {
        console.error("Webhook POST Error:", error);
        res.sendStatus(200);
    }
});

const port = 7601;
app.listen(port, () => {
    console.log(`http://localhost:${port}`)
})