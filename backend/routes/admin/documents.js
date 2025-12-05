const express = require('express');
const router = express.Router();
const connection = require('../../database/db');
const executeQuery = require('../../utils/executeQuery');
const fs = require('fs');
const { middleware } = require('../../middleware/middleware');
const multer = require('multer')
const path = require('path');
const { default: axios } = require('axios');

const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        cb(null, path.join(__dirname, '../../documents'));
    },
    filename: (req, file, cb) => {
        const { fileName } = req.body;
        if (!fileName) {
            return cb(new Error('fileName is required'), null);
        }
        cb(null, fileName + path.extname(file.originalname));
    },
});

const upload = multer({
    storage: storage,
    limits: { fileSize: 10 * 1024 * 1024 },
});

router.post('/uploadDocument', middleware, upload.single('file'), async (req, res) => {
    try {
        const { fileName } = req.body;

        if (!req.file || !fileName) {
            return res.status(400).json({ error: 'File and fileName are required.' });
        }

        const fileExtension = path.extname(req.file.originalname);
        const file_name = fileName + fileExtension;
        const [checkDocument] = await executeQuery(`select * from documents where name='${file_name}' and status='0'`)
        if (checkDocument) {
            return res.status(400).json({ error: "File with this name and extension already exists" })
        }

        connection.query('INSERT INTO documents (name) VALUES (?)', [file_name], (err, data) => {
            if (err) {
                console.log(err);
                return res.status(400).json({ error: "Something went wrong" })
            }
            return res.status(200).json({ success: 'File uploaded successfully.', data });
        });

    } catch (error) {
        console.error('/uploadDocument Error:', error.message);
        return res.status(500).json({ error: 'Internal Server Error.' });
    }
});
const Url = process.env.VITE_BOT_API_URL;

router.get("/deleteDocument", middleware, async (req, res) => {
    try {
        const { document_id } = req.query;

        if (!document_id) {
            return res.status(400).json({ error: "Query is required" })
        }
        const [document] = await executeQuery(`SELECT name FROM documents WHERE id = ${document_id} and status=0`);
        console.log(" document" + document)
        if (!document) {
            return res.status(404).json({ error: "Document not found" });
        }
        const fileName = document.name;
        const filePath = path.join(__dirname, '../../documents', fileName);
        // console.log([path.basename(filePath)])


        try {
            await fs.promises.unlink(filePath);
            connection.query(`update documents set status=1 where id=?`, [document_id], async (err, data) => {
                if (err) {
                    console.log(err);
                    return res.status(400).json({ error: "Something went wrong" })
                }
                try {
                    const pythonApiRes = await axios.post(`${Url}/remove_by_paths`, {
                        paths: [path.basename(filePath)],
                    });
                    console.log("pythonApiRes" + pythonApiRes)
                    console.log("data" + data)
                    return res.json({
                        success: "File deleted successfully.",
                        pythonService: pythonApiRes.data,
                        data,
                    });
                } catch (apiErr) {
                    console.error("Python API Error:", apiErr);
                    console.error("Python API Error:", apiErr.message);
                    return res.status(500).json({ success: "File deleted locally but failed to notify Python service." });
                }
                // return res.json({ success: "File deleted successfully.", data });
            })
        } catch (unlinkError) {
            console.log(unlinkError)
            if (unlinkError.code === 'ENOENT') {
                return res.status(404).json({ error: "File not found" });
            }
            return res.status(500).json({ error: "Failed to delete file" });
        }
    } catch (error) {
        console.error("Error in /deleteQuery:", error.message);
        return res.status(500).json({ error: "Internal Server Error" });
    }
});

router.get("/getAllDocuments", async (req, res) => {
    try {
        connection.query(`select * from documents where status=0`, (err, data) => {
            if (err) {
                console.log(err);
                return res.status(400).json({ error: "Something went wrong" })
            }
            return res.json({ success: "success", data })
        })
    } catch (error) {
        console.error("Error in /getAllQueries:", error.message);
        return res.status(500).json({ error: "Internal Server Error" });
    }
});

router.get("/getAlldocfiles", async (req, res) => {
    try {
        // const {admin_id}=req.query;
        connection.query(`select * from documents where status=0`, (err, data) => {
            if (err) {
                console.log(err);
                return res.status(400).json({ error: "Something went wrong" })
            }
            return res.json({ success: "success", data })
        })
    } catch (error) {
        console.error("Error in /getAllQueries:", error.message);
        return res.status(500).json({ error: "Internal Server Error" });
    }
});


module.exports = router;