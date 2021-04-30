const fs = require("fs")
const url = require("url")
const express = require('express')
const exec = require('child_process').exec
const app = express()
const port = 3000

project_path = 'C:\\Users\\PC\\Desktop\\Face-Adversary-Attack-Defense\\'

app.get('/facenet', (req, res) => {
    project_env = 'attack'
    pathname = url.parse(req.url).pathname;
    query = url.parse(req.url).query;
    attacker = query.split('&')[0]
    victim = query.split('&')[1]
    console.log(attacker, victim)
    content = '@echo off\n' +
    'CALL conda activate ' + project_env + '\n' +
    'CALL python ' + project_path + 'origin\\origin.py ' + attacker + ' ' + victim
    fs.writeFileSync('origin.bat', content, function(err){
        if(err) console.log('Write origin.bat error!');
        else console.log('Write origin.bat success!');
    })
    var cp = exec('origin.bat', function (error, stdout, stderr) {
        if (error) {
            console.log(error.stack);
            console.log('Error code: '+error.code);
            console.log('Signal received: '+error.signal);
        }
        console.log('stdout: ' + stdout);
        console.log('stderr: ' + stderr);
    });
    cp.on('exit', function (code) {
        console.log('子进程已退出，退出码 '+code);
        fs.exists("demo/origin/origin.png", function(exists) {
            console.log(exists ? "Display success!" : "Display fail!");
            res.writeHead(200, {'Content-Type': 'image/png' });
            // fs.createReadStream("demo/origin/origin.png", 'utf-8').pipe(res);
            var content =  fs.readFileSync("demo/origin/origin.png","binary"); 
            res.write(content,"binary"); 
            res.end();
        });
    });
})

app.get('/adv', (req, res) => {
    project_env = 'attack'
    content = '@echo off\n' + 
    'CALL conda activate ' + project_env + '\n' +
    'CALL python ' + project_path + 'adv\\fgsm.py'
    fs.writeFileSync('adv.bat', content, function(err){
        if(err) console.log('Write adv.bat error!');
        else console.log('Write adv.bat success!');
    })
    var cp = exec('adv.bat', function (error, stdout, stderr) {
        if (error) {
            console.log(error.stack);
            console.log('Error code: '+error.code);
            console.log('Signal received: '+error.signal);
        }
        console.log('stdout: ' + stdout);
        console.log('stderr: ' + stderr);
    });
    cp.on('exit', function (code) {
        console.log('子进程已退出，退出码 '+code);
        fs.exists("demo/adv/adv.png", function(exists) {
            console.log(exists ? "Display success!" : "Display fail!");
            if (exists){ 
                res.writeHead(200, {'Content-Type': 'image/png' });
                var content =  fs.readFileSync("demo/adv/adv.png","binary"); 
                res.write(content,"binary"); 
                res.end();
            }
        });
    });
})

app.get('/detect', (req, res) => {
    project_env = 'detect'
    content = '@echo off\n' + 
    'CALL conda activate ' + project_env + '\n' +
    'CALL python ' + project_path + 'detect\\detect.py'
    fs.writeFileSync('detect.bat', content, function(err){
        if(err) console.log('Write detect.bat error!');
        else console.log('Write detect.bat success!');
    })
    var cp = exec('detect.bat', function (error, stdout, stderr) {
        if (error) {
            console.log(error.stack);
            console.log('Error code: '+error.code);
            console.log('Signal received: '+error.signal);
        }
        console.log('stdout: ' + stdout);
        console.log('stderr: ' + stderr);
    });
    cp.on('exit', function (code) {
        console.log('子进程已退出，退出码 '+code);
        fs.exists("demo/detect/detect.png", function(exists) {
            console.log(exists ? "Display success!" : "Display fail!");
            if (exists){ 
                res.writeHead(200, {'Content-Type': 'image/png' });
                var content =  fs.readFileSync("demo/detect/detect.png","binary"); 
                res.write(content,"binary"); 
                res.end();
            }
        });
    });
})

app.listen(port, () => console.log(`Example app listening on port ${port}!`))

// http://127.0.0.1:3000/facenet?Elon_Musk&Jeff_Bezos
// http://127.0.0.1:3000/adv
// http://127.0.0.1:3000/detect