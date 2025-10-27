const express = require("express");
const connection = require("../../database/db.js");
const bcrypt = require("bcryptjs");
const jwt = require("jsonwebtoken");
const { middleware } = require("../../middleware/middleware");
const executeQuery = require('../../utils/executeQuery');
const { addUserSchema, updateUserSchema, deleteUserSchema } = require("../../validation/user.js");



const router = express.Router();


router.post("/login", async (req, res) => {
    try {
        const { email, password } = req.body;

        const checkEmail = await executeQuery(`select * from user where email=?`, [email]);

        if (!checkEmail[0]?.password) {
            return res.status(400).json({ error: "Please set your password" })
        }

        if (!checkEmail[0] || !password) {
            return res.status(400).json({ error: "Invalid Credentials" });
        }
        const pwdCompare = await bcrypt.compare(password, checkEmail[0].password);

        if (!pwdCompare) {
            return res.status(400).json({ error: "Invalid Credentials" })
        }

        if (pwdCompare) {
            const payload = {
                email: email,
                user_id: checkEmail[0].id,
                user_type: checkEmail[0].role
            };
            let auth_token = jwt.sign(payload, process.env.JWT_SECRET);
            await executeQuery(`update user set last_login=NOW() where id=${checkEmail[0]?.id};`)
            return res.json({ success: `Welcome Back, ${checkEmail[0]?.name}`, data: { name: checkEmail[0]?.name, email: checkEmail[0]?.email, role: checkEmail[0].role, id: checkEmail[0].id }, auth_token })
        } else {
            return res.status(400).json({ error: "Invalid Credentials." });
        }
    } catch (error) {
        console.log("auth/user/login: ", error.message);
        return res.status(500).json({ error: "Internal Server Error." });
    }
});

router.post("/signUp", async (req, res) => {
    try {
        const { name, email, password } = req.body;

        console.log(req.body);


        var salt = bcrypt.genSaltSync(10);
        const secPass = await bcrypt.hash(password, salt);
        console.log(secPass);



        const { error } = addUserSchema.validate(req.body, { abortEarly: false });

        console.log(error);

        if (error) {
            return res.status(400).json({ error: error.details[0]?.message });
        }

        const [checkEmail] = await executeQuery(`select * from user where email=?`, [email])
        if (checkEmail) {
            return res.status(400).json({ error: "Email already exist" })
        }

        console.log(checkEmail);

        const insertQuery = 'insert into user (name, email,password) values (?, ?, ?);'
        connection.execute(insertQuery, [name, email, secPass], (err, data) => {
            if (err) {
                // console.log(err);
                return res.status(400).json({ error: "Something went wrong" })
            }
            return res.json({ success: "User Added", data })
        })
    } catch (error) {
        console.log("auth/addUser: ", error.message);
        return res.status(500).json({ error: "Internal Server Error." });
    }
});

router.post("/updateUser", async (req, res) => {
    try {
        const { user_id, password, ...rest } = req.body;
        const { error } = updateUserSchema.validate(req.body, { abortEarly: false });
        if (error) {
            return res.status(400).json({ error: error.details.map(err => err.message) });
        }

        if (password) {
            const salt = bcrypt.genSaltSync(10);
            const secPass = await bcrypt.hash(password, salt);
            rest.password = secPass; // Replace plaintext password with hashed password
        }

        const fields = Object.keys(rest);
        const values = Object.values(rest);

        if (fields.length === 0) {
            return res.status(400).json({ error: "No valid fields provided for update." });
        }

        const setClause = fields.map(field => `${field} = ?`).join(", ");
        values.push(user_id);

        const query = `UPDATE user SET ${setClause}, updated_at = CURRENT_TIMESTAMP WHERE id = ?`;

        connection.execute(query, values, (err, data) => {
            if (err) {
                console.log(err);
                return res.status(400).json({ error: "Something went wrong" })
            }
            if (data.affectedRows === 0) {
                return res.status(404).json({ error: "Record not found" });
            }
            return res.json({ success: "User updated", data })
        });
    } catch (error) {
        console.error("Error in /updateUser :", error.message);
        return res.status(500).json({ error: "Internal Server Error" });
    }
});

router.get("/deleteUser", async (req, res) => {
    try {
        const { user_id } = req.query;

        const { error } = deleteUserSchema.validate(req.query, { abortEarly: false });
        if (error) {
            return res.status(400).json({ error: error.details.map(err => err.message) });
        }

        const query = `UPDATE user SET status=1, updated_at = CURRENT_TIMESTAMP WHERE id = ?`;

        connection.execute(query, [user_id], (err, data) => {
            if (err) {
                console.log(err);
                return res.status(400).json({ error: "Something went wrong" })
            }
            if (data.affectedRows === 0) {
                return res.status(404).json({ error: "Record not found" });
            }
            return res.json({ success: "User deleted", data })
        });
    } catch (error) {
        console.error("Error in /updateUser:", error.message);
        return res.status(500).json({ error: "Internal Server Error" });
    }
});

router.get("/getAllUser", async (req, res) => {
    try {
        connection.query(`select id, name, email, mobile_no, status, created_at, updated_at, last_login, date_of_birth, account_status from user where status=0`, (err, result) => {
            if (err) {
                console.log(err);
                return res.status(400).json({ error: "Something went wrong" })
            }
            return res.json({ success: "success", data: result })
        })
    } catch (error) {
        console.error("Error in /getAllUser:", error.message);
        return res.status(500).json({ error: "Internal Server Error" });
    }
});

router.get("/getUser/:id", async (req, res) => {
    try {
        const { id } = req.params;
        connection.query(`select * from user where id=${id}`, (err, result) => {
            if (err) {
                console.log(err);
                return res.status(400).json({ error: "Something went wrong" })
            }
            return res.json({ success: "success", data: result })
        })
    } catch (error) {
        console.error("Error in /getAllUser:", error.message);
        return res.status(500).json({ error: "Internal Server Error" });
    }
});

module.exports = router;