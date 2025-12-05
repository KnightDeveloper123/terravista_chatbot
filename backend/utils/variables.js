const BOT_URL = `http://216.10.251.154:5000`

function formatForWhatsApp(text) {
    if (!text) return "";

    let msg = String(text);

    // Tabs → spaces
    msg = msg.replace(/\t/g, "    ");

    // 1️⃣ Convert HTML tags to plain text
    msg = msg.replace(/<\/?[^>]+>/g, "");

    // 2️⃣ Convert **bold** → ALL CAPS + bold → *TEXT*
    msg = msg.replace(/\*\*(.+?)\*\*/gs, (_, inner) => {
        return `*${inner.trim().toUpperCase()}*`;
    });

    // 3️⃣ Convert *text* → ALL CAPS + bold → *TEXT*
    // Avoid conflict with **...** already processed
    msg = msg.replace(/(^|[^*])\*(?!\*)([^*\n]+)\*(?!\*)/g, (_, prefix, inner) => {
        return `${prefix}*${inner.trim().toUpperCase()}*`;
    });

    // 4️⃣ Markdown links [label](url) → url
    msg = msg.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, "$2");

    // 5️⃣ Remove multiple spaces
    msg = msg.replace(/ +/g, " ");

    return msg.trim();
}

module.exports = {
    BOT_URL,
    formatForWhatsApp
}