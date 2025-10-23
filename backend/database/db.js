var mysql2 = require('mysql2')
var connection = mysql2.createPool({
    host: 'localhost',
    user: 'root',
    password: 'root',
    database: 'cipla_chatbot',
})


connection.getConnection((err) => {
    if (err) {
        console.log(err)
        return
    }
    console.log('Database connected')
})



module.exports = connection;