const express = require("express");
const cors = require("cors");
require('dotenv').config()
const path = require("path");



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


const port = 7601;
app.listen(port, () => {
    console.log(`http://localhost:${port}`)
})