const express = require('express');
const router = express.Router();
const connection = require('../../database/db');
const executeQuery = require('../../utils/executeQuery');
const { addEmployeeSchema, updateEmployeeSchema, deleteEmployeeSchema } = require("../../validation/employee");
const { sendOtp } = require('../../utils/mail')
const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');
const { middleware } = require('../../middleware/middleware');
const fs = require('fs');
const multer = require('multer')
const path = require('path')
const crypto = require('crypto');
const { data } = require('react-router-dom');

const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        cb(null, path.join(__dirname, '../../profile'));
    },
    filename: (req, file, cb) => {

        const uniqueName = `${Date.now()}-${Math.round(Math.random() * 1E9)}${path.extname(file.originalname)}`;
        cb(null, uniqueName);
    },
});

const upload = multer({
    storage: storage,
    limits: { fileSize: 10 * 1024 * 1024 },
});


router.post("/add", async (req, res) => {
    try {
        const { name, email, mobile_no, date_of_birth, role } = req.body;

        const { error } = addEmployeeSchema.validate(req.body, { abortEarly: false });

        if (error) {
            return res.status(400).json({ error: error.details[0]?.message });
        }

        const [checkEmail] = await executeQuery(`select * from admin where email=?`, [email])
        if (checkEmail) {
            return res.status(400).json({ error: "Email already exist" })
        }

        const insertQuery = 'insert into admin (name, email, mobile_no, date_of_birth, role) values (?, ?, ?, ?, ? );'
        connection.execute(insertQuery, [name, email, mobile_no, date_of_birth ?? null, role], (err, data) => {
            if (err) {
                console.log(err);
                return res.status(400).json({ error: "Something went wrong" })
            }
            return res.json({ success: "admin Added", data })
        })
    } catch (error) {
        console.log("/add: ", error.message);
        return res.status(500).json({ error: "Internal Server Error." });
    }
});

router.post("/update", middleware, async (req, res) => {
    try {
        const { employee_id, ...rest } = req.body;
        const { error } = updateEmployeeSchema.validate(req.body, { abortEarly: false });
        if (error) {
            return res.status(400).json({ error: error.details.map(err => err.message) });
        }

        const fields = Object.keys(rest);
        const values = Object.values(rest);

        if (fields.length === 0) {
            return res.status(400).json({ error: "No valid fields provided for update." });
        }

        const setClause = fields.map(field => `${field} = ?`).join(", ");
        values.push(employee_id);

        const query = `UPDATE admin SET ${setClause}, updated_at = CURRENT_TIMESTAMP WHERE id = ?`;

        connection.execute(query, values, (err, data) => {
            if (err) {
                console.log(err);
                return res.status(400).json({ error: "Something went wrong" })
            }
            if (data.affectedRows === 0) {
                return res.status(404).json({ error: "Record not found" });
            }
            return res.json({ success: "admin updated", data })
        });
    } catch (error) {
        console.error("Error in /update :", error.message);
        return res.status(500).json({ error: "Internal Server Error" });
    }
});

router.post("/delete", middleware, async (req, res) => {
    try {
        const { user_id } = req.body;

        const { error } = deleteEmployeeSchema.validate(req.body, { abortEarly: false });
        if (error) {
            return res.status(400).json({ error: error.details.map(err => err.message) });
        }

        const query = `UPDATE admin SET status=1, updated_at = CURRENT_TIMESTAMP WHERE id = ?`;

        connection.execute(query, [user_id], (err, data) => {
            if (err) {
                console.log(err);
                return res.status(400).json({ error: "Something went wrong" })
            }
            if (data.affectedRows === 0) {
                return res.status(404).json({ error: "Record not found" });
            }
            return res.json({ success: "admin deleted", data })
        });
    } catch (error) {
        console.error("Error in /update:", error.message);
        return res.status(500).json({ error: "Internal Server Error" });
    }
});

router.get('/getEmployeeId', middleware, async (req, res) => {
    try {
        const { user_id } = req.query;
        const data = await executeQuery(`select * from admin where id=${user_id}`)
        return res.json({ data: data[0] })
    } catch (error) {
        console.log(error)
        return res.status(500).json({ error: "Internal Server Error" })
    }
});

router.post("/update", middleware, upload.single('profile'), async (req, res) => {
    try {
        const { employee_id, ...rest } = req.body;

        const fields = Object.keys(rest);
        let values = Object.values(rest).map(val => val === undefined ? null : val);

        // If a profile picture is uploaded
        if (req.file) {
            fields.push('profile'); // <-- VERY IMPORTANT
            values.push(req.file.filename);
        }

        if (fields.length === 0) {
            return res.status(400).json({ error: "No valid fields provided for update." });
        }

        const setClause = fields.map(field => `${field} = ?`).join(", ");

        values.push(employee_id); // employee_id is added last, for WHERE id = ?

        const query = `UPDATE admin SET ${setClause}, updated_at = CURRENT_TIMESTAMP WHERE id = ?`;

        connection.query(query, values, (err, data) => {
            if (err) {
                console.log('MYSQL ERROR:', err);
                return res.status(400).json({ error: "Something went wrong" });
            }
            console.log('MYSQL SUCCESS:', data);
            if (data.affectedRows === 0) {
                return res.status(404).json({ error: "Record not found" });
            }
            return res.json({ success: "admin updated", data });
        });
    } catch (error) {
        console.error("Error in /update:", error.message);
        return res.status(500).json({ error: "Internal Server Error" });
    }
});

let otpStore = {}
router.post("/login", async (req, res) => {
    try {
        // console.log(req.body);

        const { email, password } = req.body;

        const user = await executeQuery(`select * from admin where email=?`, [email]);
        // console.log(user);
        if (!user[0] || !password) {
            return res.status(400).json({ error: "Invalid Credentials" });
        }


        if (!user[0]?.password) {
            return res.status(400).json({ error: "Please set your password" });
        }

        const pwdCompare = await bcrypt.compare(password, user[0].password);
        // console.log(pwdCompare);


        if (!pwdCompare) {
            return res.status(400).json({ error: "Invalid Credentials" });
        }

        // ✅ BLOCK inactive users first
        if (user[0].is_active === 1) {
            return res.status(403).json({ error: "Your account is inactive. Please contact Super Admin." });
        }

        if (user[0].role === 'Admin') {
            const otp = crypto.randomInt(100000, 999999).toString();
            const expiry = Date.now() + 5 * 60 * 1000;
            otpStore[email] = { otp, expiry };

            await sendOtp(otp, email, user[0].name);

            return res.json({ step: "otp-verification", message: "OTP sent to email", email });
        }

        // ✅ Normal login for other roles
        const payload = {
            email: email,
            user_id: user[0].id,
            user_type: user[0].role
        };
        let auth_token = jwt.sign(payload, process.env.JWT_SECRET);
        await executeQuery(`update admin set last_login=NOW() where id=?`, [user[0].id]);

        return res.json({
            success: `Welcome Back, ${user[0].name}`,
            data: {
                name: user[0].name,
                email: user[0].email,
                role: user[0].role,
                id: user[0].id
            },
            auth_token
        });

    } catch (error) {
        console.log("/login: ", error.message);
        return res.status(500).json({ error: "Internal Server Error." });
    }
});


router.post('/verify-otp', async (req, res) => {

    const { email, otp } = req.body;

    // if (!email || !otp) {
    //     return res.status(400).json({ error: "Email and OTP are required" });
    //   }

    if (!otpStore[email]) return res.status(400).json({ error: "OTP  not requested" })

    const { otp: storedOtp, expiry } = otpStore[email];

    if (Date.now() > expiry) return res.status(400).json({ error: "OTP expired" })
    if (otp === storedOtp) {

        const rows = await executeQuery(`SELECT * FROM admin WHERE email = ?`, [email]);
        const user = rows[0];

        if (!user) {
            return res.status(404).json({ error: "User not found" });
        }
        delete otpStore[email]

        await executeQuery(`UPDATE admin SET last_login = NOW() WHERE id = ?`, [user.id]);

        const payload = { email, user_id: user.id, user_type: user.role };
        const token = jwt.sign(payload, process.env.JWT_SECRET);

        return res.json({
            success: `Welcome back, ${user.name}`,
            data: { name: user.name, email: user.email, role: user.role, id: user.id },
            auth_token: token
        });
        // res.status(200).json({ success: "OTP verified successfully" })
    } else {
        res.status(400).json({ error: "Invalid OTP" })
    }

})


router.post("/signUp", async (req, res) => {
    try {
        const { name, email, password } = req.body;

        var salt = bcrypt.genSaltSync(10);
        const secPass = await bcrypt.hash(password, salt);
        // const { error } = addUserSchema.validate(req.body, { abortEarly: false });

        // if (error) {
        //     return res.status(400).json({ error: error.details[0]?.message });
        // }

        const [checkEmail] = await executeQuery(`select * from admin where email=?`, [email])
        if (checkEmail) {
            return res.status(400).json({ error: "Email already exist" })
        }


        const insertQuery = 'insert into admin (name, email,password, role) values (?, ?, ?, ?);'
        connection.execute(insertQuery, [name, email, secPass, 'Admin'], (err, data) => {
            if (err) {
                // console.log(err);
                return res.status(400).json({ error: "Something went wrong" })
            }
            return res.json({ success: "Registration successfully", data })
        })
    } catch (error) {
        return res.status(500).json({ error: "Internal Server Error." });
    }
});

// router.post("/changePassword", async (req, res) => {
//     try {
//         const { email, password } = req.body;

//         const checkEmail = await executeQuery(`select * from admin where email=?`, [email]);

//         if (checkEmail[0]) {
//             console.log("its check email ");

//             const otp = crypto.randomInt(100000, 999999).toString();
//             const expiry = Date.now() + 5 * 60 * 1000;
//             otpStore[email] = { otp, expiry };

//             await sendOtp(otp, email, checkEmail[0].name);

//              return res.json({ step: "otp-verification", message: "OTP sent to email", email });
//         }


//         var salt = bcrypt.genSaltSync(10);
//         const secPass = await bcrypt.hash(password, salt);

//         connection.execute('update admin set password=? where email=?;', [secPass, email], (err, data) => {
//             if (err) {
//                 console.log(err);
//                 return res.status(400).json({ error: "Something went wrong" })
//             }
//             return res.json({ success: `Password changed`, data })
//         })

//     } catch (error) {
//         console.log("/changePassword: ", error.message);
//         return res.status(500).json({ error: "Internal Server Error." });
//     }
// });

router.post("/changePassword", async (req, res) => {
    try {
        const { email, password, otp, step } = req.body;

        const checkEmail = await executeQuery(`SELECT * FROM admin WHERE email=?`, [email]);

        if (!checkEmail[0]) {
            return res.status(404).json({ error: "Email not found" });
        }

        if (step === "request-otp") {
            const generatedOtp = crypto.randomInt(100000, 999999).toString();
            const expiry = Date.now() + 5 * 60 * 1000; // 5 minutes
            otpStore[email] = { otp: generatedOtp, expiry };
            log
            await sendOtp(generatedOtp, email, checkEmail[0].name);
            return res.json({ step: "otp-verification", message: "OTP sent to email", email });
        }

        if (step === "verify-and-reset") {
            if (!otp || !otpStore[email]) {
                return res.status(400).json({ error: "OTP not found or expired" });
            }

            const { otp: storedOtp, expiry } = otpStore[email];
            if (Date.now() > expiry || otp !== storedOtp) {
                return res.status(400).json({ error: "Invalid or expired OTP" });
            }

            // Proceed to change password
            const salt = bcrypt.genSaltSync(10);
            const secPass = await bcrypt.hash(password, salt);

            connection.execute('UPDATE admin SET password=? WHERE email=?;', [secPass, email], (err, data) => {
                if (err) {
                    console.log(err);
                    return res.status(400).json({ error: "Something went wrong" });
                }

                // Clean up OTP
                delete otpStore[email];

                return res.status(200).json({ success: "Password changed", data });
            });
        } else {
            return res.status(400).json({ error: "Invalid step" });
        }

    } catch (error) {
        console.log("/changePassword: ", error.message);
        return res.status(500).json({ error: "Internal Server Error." });
    }
});

router.get("/getAllEmployee", middleware, async (req, res) => {
    try {
        connection.query(`select * from admin where status=0`, (err, result) => {
            if (err) {
                console.log(err);
                return res.status(400).json({ error: "Something went wrong" })
            }
            console.log("result", result)
            return res.json({ success: "success", data: result })
        })
    } catch (error) {
        console.error("Error in /getAllEMployee:", error.message);
        return res.status(500).json({ error: "Internal Server Error" });
    }
});

router.get("/getEmployeeById", middleware, async (req, res) => {
    try {
        const { employee_id } = req.query;

        connection.query(`select * from admin where status=0 and id=?;`, [employee_id], (err, result) => {
            if (err) {
                return res.status(500).json({ error: "Internal Server Error" });
            }
            return res.json({ success: "success", data: result[0] })
        })
    } catch (error) {
        console.error("Error in /getEmployeeById:", error.message);
        return res.status(500).json({ error: "Internal Server Error" });
    }
});

router.get('/getAllDashboardData', middleware, async (req, res) => {
    try {
        connection.query(`SELECT 
                COUNT(*) AS total_employee,
                (SELECT COUNT(*) FROM admin WHERE is_active = 0) AS total_admin,
                (SELECT COUNT(*) FROM user WHERE status = 0) AS total_user,
                (SELECT COUNT(*) FROM support WHERE query_status = 'pending') AS pending_queries
            FROM employee
            WHERE status = 0;
            `,
            (err, result) => {
                if (err) {
                    console.log(err);
                    return res.status(400).json({ error: "Something went wrong" })
                }
                // console.log(result)
                return res.json({ success: "success", counts: result[0] })
            })
    } catch (error) {
        console.error("Error in /getAllQueriesById:", error.message);
        return res.status(500).json({ error: "Internal Server Error" });
    }
})

router.post("/getActiveStatus", middleware, async (req, res) => {
    try {
        const { user_id, is_active } = req.body;

        const query = `UPDATE admin SET is_active=${is_active}, updated_at = CURRENT_TIMESTAMP WHERE id = ?`;

        connection.execute(query, [user_id], (err, data) => {
            if (err) {
                console.log(err);
                return res.status(400).json({ error: "Something went wrong" })
            }
            if (data.affectedRows === 0) {
                return res.status(404).json({ error: "Record not found" });
            }
            return res.json({ success: "User is Active", data })
        });
    } catch (error) {
        console.error("Error in /getActiveStatus:", error.message);
        return res.status(500).json({ error: "Internal Server Error" });
    }
});


router.get('/getAdminCount', middleware, async (req, res) => {
    try {
        // Get count of active bots in the last 30 days
        const { admin_id } = req.query
        const activeBot = await executeQuery(`
      SELECT COUNT(*) AS active_bots FROM bots 
      WHERE status=0 AND admin_id=${admin_id} AND created_at >= NOW() - INTERVAL 30 DAY
    `);

        // Get count of campaigns with status 'Sent'
        const campaignSent = await executeQuery(`
            SELECT 
        c.admin_id,
        COUNT(ml.id) AS total_messages_sent
        FROM 
        messages_log ml
        JOIN 
        campaign c ON ml.campaign_id = c.id
        WHERE 
        ml.status = 'sent' AND c.admin_id=${admin_id} IS NOT NULL
        GROUP BY 
        c.admin_id;

    `);
        // console.log(activeBot)

        // Send response in the expected format
        return res.json({
            success: true,
            data: {
                activeBots: activeBot[0].active_bots,
                campaignsSent: campaignSent[0].total_messages_sent
            }
        });

    } catch (error) {
        console.error("Error in /getAdminCount:", error.message);
        return res.status(500).json({ success: false, error: "Internal Server Error" });
    }
});


// month wise display bot
router.get('/getMonthlyMetrics', middleware, async (req, res) => {
    try {
        const botsQuery = `
      SELECT DATE_FORMAT(created_at, '%Y-%m') AS month, COUNT(*) AS count
      FROM bots
      GROUP BY month
      ORDER BY month;
    `;

        const campaignsQuery = `
      SELECT DATE_FORMAT(created_at, '%Y-%m') AS month, COUNT(*) AS count
      FROM campaign
      WHERE is_status = 'Sent'
      GROUP BY month
      ORDER BY month;
    `;

        const [bots, campaigns] = await Promise.all([
            executeQuery(botsQuery),
            executeQuery(campaignsQuery)
        ]);

        // get totals
        const totalActiveBots = bots.reduce((sum, row) => sum + row.count, 0);
        const totalCampaignsSent = campaigns.reduce((sum, row) => sum + row.count, 0);

        res.json({
            success: true,
            data: {
                totalActiveBots,
                totalCampaignsSent,
                botsByMonth: bots,
                campaignsByMonth: campaigns
            }
        });
    } catch (err) {
        console.error('Error fetching metrics:', err);
        res.status(500).json({ error: 'Internal server error' });
    }
});

//sector-wise bots
router.get('/sector-performance', middleware, async (req, res) => {
    try {
        const result = await executeQuery(`
      SELECT s.name AS sector, COUNT(b.id) AS total_bots
      FROM bots b
      JOIN sector s ON s.id = b.sector_id
      GROUP BY b.sector_id
    `);

        return res.json({ success: true, data: result });
    } catch (err) {
        console.error(err);
        return res.status(500).json({ error: "Internal Server Error" });
    }
});

// sector-wise number of bots
router.get('/sectorNumberWiseBots', middleware, async (req, res) => {
    const { admin_id } = req.query
    try {
        const result = await executeQuery(`SELECT s.name AS sector_name, COUNT(*) AS bot_count
            FROM bots b
            JOIN sector s ON b.sector_id = s.id
            WHERE b.status = 0 AND b.admin_id = ${admin_id}
            GROUP BY s.name;
            ;
         `);

        const generativeData = await executeQuery(`SELECT s.name AS sector_name, COUNT(*) AS bot_count
        FROM documents d
        JOIN sector s ON d.sector_id = s.id
        WHERE d.bot_type = 'Genarative ai'
        AND d.status = 0
        AND d.admin_id =  ${admin_id}
        GROUP BY s.name;
        `)


        return res.json({ success: true, data: result, generativeData: generativeData });
    } catch (err) {
        console.error(err);
        return res.status(500).json({ error: "Internal Server Error" });
    }
});

// user_engagement
// router.get('/user-engagement', middleware, async (req, res) => {
//   try {
//     const ctr = await executeQuery(`
//       SELECT ROUND(SUM(clicks)/SUM(impressions)*100, 2) AS click_through_rate FROM user_metrics
//     `);

//     const completionRate = await executeQuery(`
//       SELECT ROUND(SUM(completed)/SUM(started)*100, 2) AS completion_rate FROM user_metrics
//     `);

//     return res.json({
//       success: true,
//       data: {
//         clickThroughRate: ctr[0].click_through_rate || 0,
//         completionRate: completionRate[0].completion_rate || 0,
//       }
//     });
//   } catch (err) {
//     console.error(err);
//     return res.status(500).json({ error: "Internal Server Error" });
//   }
// });


//top performing bots and campaign
// router.get('/top-performing', middleware, async (req, res) => {
//     try {
//         const bots = await executeQuery(`
//       SELECT id, name, total_users
//       FROM bots
//       ORDER BY total_users DESC
//       LIMIT 5
//     `);

//         const campaigns = await executeQuery(`
//       SELECT id, campaign_name, clicks
//       FROM campaign
//       ORDER BY clicks DESC
//       LIMIT 5
//     `);

//         return res.json({
//             success: true,
//             data: {
//                 topBots: bots,
//                 topCampaigns: campaigns
//             }
//         });
//     } catch (err) {
//         console.error(err);
//         return res.status(500).json({ error: "Internal Server Error" });
//     }
// });

router.get('/top-performing-bots', async (req, res) => {
    try {
        const topBots = await executeQuery(`
      SELECT 
        b.id AS bot_id,
        b.name AS bot_name,
         
        COUNT(bu.user_id) AS total_users
      FROM bots b
      LEFT JOIN bot_users bu ON b.id = bu.bot_id
      GROUP BY b.id, b.name
      ORDER BY total_users DESC
      LIMIT 5;
    `);

        res.json({
            success: true,
            data: topBots
        });
    } catch (err) {
        console.error(err);
        res.status(500).json({ success: false, message: 'Internal server error' });
    }
});

router.get('/top-performing-campaigns', async (req, res) => {
    try {
        const topBots = await executeQuery(`
      SELECT 
        c.id AS campaign_id,
        c.campaign_name AS name,
         
        COUNT(cu.user_id) AS total_users
      FROM campaign c
      LEFT JOIN campaign_users cu ON c.id = cu.campaign_id
      GROUP BY c.id, c.campaign_name
      ORDER BY total_users DESC
      LIMIT 5;
    `);

        res.json({
            success: true,
            data: topBots
        });
    } catch (err) {
        console.error(err);
        res.status(500).json({ success: false, message: 'Internal server error' });
    }
});


// using clicks rate bot id,admin-id,campiagan_id
router.get('/botcampaigns-by-clicks', async (req, res) => {
    try {
        const topClickedCampaigns = await executeQuery(`SELECT
    b.admin_id,
    COUNT(DISTINCT il_submit.id) AS completed_interactions,
    COUNT(DISTINCT il_click.id) AS total_clicks,
    COUNT(DISTINCT ml.id) AS total_messages,
    ROUND((COUNT(DISTINCT il_submit.id) / NULLIF(COUNT(DISTINCT ml.id), 0)) * 100, 2) AS completion_rate,
    ROUND((COUNT(DISTINCT il_click.id) / NULLIF(COUNT(DISTINCT ml.id), 0)) * 100, 2) AS click_through_rate
FROM bots b
LEFT JOIN messages_log ml ON b.id = ml.bot_id
LEFT JOIN interactions_log il_submit ON b.id = il_submit.bot_id AND il_submit.interaction_type = 'submit'
LEFT JOIN interactions_log il_click ON b.id = il_click.bot_id AND il_click.interaction_type = 'click'
GROUP BY b.admin_id;

 `);
        res.json({
            success: true,
            data: topClickedCampaigns
        });
        // console.log(topClickedCampaigns)
    } catch (err) {
        console.error('Error fetching top campaigns by clicks:', err);
        res.status(500).json({ success: false, message: 'Internal server error' });
    }
});


router.get('/getActiveUser', middleware, async (req, res) => {
    try {
        const [daily, monthly] = await Promise.all([
            executeQuery(`SELECT DATE(last_login) AS day, COUNT(DISTINCT id) AS daily_active_users
                    FROM admin
                    WHERE last_login >= CURDATE() - INTERVAL 30 DAY
                    GROUP BY day ORDER BY day;`),

            executeQuery(`SELECT DATE_FORMAT(last_login, '%Y-%m') AS month, COUNT(DISTINCT id) AS monthly_active_users
                    FROM admin
                    WHERE last_login >= CURDATE() - INTERVAL 12 MONTH
                    GROUP BY month ORDER BY month;`)
        ]);
        // console.log(daily)
        // console.log(monthly)
        res.json({ success: true, daily, monthly });
    } catch (err) {
        console.error('Error fetching active users:', err);
        res.status(500).json({ success: false, message: 'Internal server error' });
    }
});





module.exports = router;