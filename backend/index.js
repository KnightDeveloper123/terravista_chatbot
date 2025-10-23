require("dotenv").config();
const express = require("express");
const cors = require("cors");
require('dotenv').config()
const path = require("path");



const app = express();

// const allowedOrigins = [process.env.FRONTEND_URL];

// app.use(cors({
//     origin: function (origin, callback) {
//         if (!origin || allowedOrigins.includes(origin)) {
//             callback(null, true);
//         } else {
//             callback(new Error("CORS Not Allowed"));
//         }
//     },
//     methods: ["GET", "POST", "PUT", "DELETE"],
//     allowedHeaders: ["Content-Type", "Authorization"],
//     credentials: true
// }));
// app.use(cors());

app.use(cors({
    origin: "http://localhost:5173", // 👈 your frontend URL
    credentials: true,               // 👈 allow cookies / auth headers
    methods: ["GET", "POST", "PUT", "DELETE"],
}));

app.use(express.json());

app.use("/documents", express.static(path.join(__dirname, "documents/")));
app.use("/products", express.static(path.join(__dirname, "products/")));
app.use("/sectors", express.static(path.join(__dirname, "sectors/")));
app.use("/profile", express.static(path.join(__dirname, "profile/")));
app.use("/uploads", express.static(path.join(__dirname, "uploads/")));
app.use("/uploads/community", express.static(path.join(__dirname, "uploads/community")));

app.use('/uploadFiles', express.static(path.join(__dirname, 'uploadFiles/')));
app.use('/videoFiles', express.static(path.join(__dirname, 'videoFiles/')));

// app.use('/uploadFiles', express.static(path.join(__dirname, 'uploadFiles')));

app.use(express.urlencoded({ extended: true }));
app.use(express.json());

app.use('/admin', require("./routes/admin/admin"))
app.use('/support', require("./routes/admin/support"))
app.use('/documents', require("./routes/admin/documents"))
app.use('/product_service', require("./routes/admin/product_service"))
app.use('/campaign', require("./routes/admin/campaign"))
app.use('/sector', require("./routes/admin/sector"))
app.use('/employee', require("./routes/admin/employee"))
app.use('/contact', require("./routes/admin/contact_list"))
app.use('/bots', require("./routes/admin/bots"))
app.use('/template', require("./routes/admin/template"))
app.use('/', require('./routes/admin/webhook'))

app.use('/user', require("./routes/users/user"))
app.use('/community', require("./routes/users/community"))

app.use('/chatbot', require("./routes/users/chatbot"))
// app.use('/webhook', require('./routes/admin/bots'));


const port = 2500;
app.listen(port, () => {
    console.log(`http://localhost:${port}`)

})