const express = require('express')
const app = express()
const port = 3010;
const model = require('./TSC_json_model.json')

app.use(function(req, res, next) {
    res.header("Access-Control-Allow-Origin", "*"); // update to match the domain you will make the request from
    res.header("Access-Control-Allow-Headers", "Origin, X-Requested-With, Content-Type, Accept");
    res.header('Content-Type', 'application/json');
    next();
  });


app.get('/getModel', (req, res) => {
  res.header("Content-Type", 'application/json');
  res.send(model);
})

app.listen(port, () => {
  console.log("running on port: ", port);
})