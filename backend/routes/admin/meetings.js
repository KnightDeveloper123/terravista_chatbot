const express = require('express');
const router = express.Router();
const executeQuery = require('../../utils/executeQuery');
const { middleware } = require('../../middleware/middleware');

router.post('/scheduleMeeting', async (req, res) => {
    try {
        const { user_id, meeting_date, description } = req.body;

        if (!user_id || !meeting_date) {
            return res.status(400).json({ error: 'user_id and meeting_date are required.' });
        }

        const insertQuery = `INSERT INTO meetings (user_id, meeting_date, description, status) VALUES (?, ?, ?, 'pending')`;
        const meeting = await executeQuery(insertQuery, [user_id, meeting_date, description]);

        return res.status(201).json({ success: 'Meeting scheduled successfully. Admin will add the meeting link.', meeting });

    } catch (error) {
        console.error('/scheduleMeeting Error:', error.message);
        return res.status(500).json({ error: 'Internal Server Error.' });
    }
});

router.get('/getMeetings', middleware, async (req, res) => {
    try {
        const meetings = await executeQuery(`SELECT * FROM meetings ORDER BY meeting_date DESC`);

        return res.status(200).json({ success: 'Meetings fetched successfully', data: meetings });
    } catch (error) {
        console.error('/getMeetings Error:', error.message);
        return res.status(500).json({ error: 'Internal Server Error.' });
    }
});

router.put('/updateMeeting', middleware, async (req, res) => {
    try {
        const { meeting_id, meeting_link, status } = req.body;

        if (!meeting_id) {
            return res.status(400).json({ error: 'meeting_id is required.' });
        }

        const updates = [];
        const values = [];

        if (meeting_link) {
            updates.push(`meeting_link = ?`);
            values.push(meeting_link);
        }

        if (status) {
            updates.push(`status = ?`);
            values.push(status);
        }

        if (updates.length === 0) {
            return res.status(400).json({ error: 'No fields to update.' });
        }

        values.push(meeting_id);

        const updateQuery = `UPDATE meetings SET ${updates.join(', ')}, updated_at = NOW() WHERE id = ? RETURNING *;`;
        const updatedMeeting = await executeQuery(updateQuery, values);

        return res.status(200).json({ message: 'Meeting updated successfully' });
    } catch (error) {
        console.error('/updateMeeting Error:', error.message);
        return res.status(500).json({ error: 'Internal Server Error.' });
    }
});

module.exports = router;