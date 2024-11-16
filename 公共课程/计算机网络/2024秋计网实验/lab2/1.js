// ==UserScript==
// @name         雨课堂答案显示和pdf下载
// @namespace    http://tampermonkey.net/
// @version      1.6.0
// @description  雨课堂答案显示
// @author       文程
// @match          *://*.yuketang.cn/*
// @grant	       GM_getValue
// @grant	       GM_setValue
// @grant	       GM_registerMenuCommand
// @grant	       GM_addStyle
// @grant	       GM.setClipboard
// @require      https://cdnjs.cloudflare.com/ajax/libs/jspdf/1.3.2/jspdf.debug.js
// @license MIT

// @downloadURL https://update.greasyfork.org/scripts/462467/%E9%9B%A8%E8%AF%BE%E5%A0%82%E7%AD%94%E6%A1%88%E6%98%BE%E7%A4%BA%E5%92%8Cpdf%E4%B8%8B%E8%BD%BD.user.js
// @updateURL https://update.greasyfork.org/scripts/462467/%E9%9B%A8%E8%AF%BE%E5%A0%82%E7%AD%94%E6%A1%88%E6%98%BE%E7%A4%BA%E5%92%8Cpdf%E4%B8%8B%E8%BD%BD.meta.js
// ==/UserScript==

(function () {
    'use strict';
    const e = document.createElement("script");
    e.src = chrome.runtime.getURL("injected.js"),
        e.onload = function () {
            e.remove()
        }
        ,
        (document.head || document.documentElement).appendChild(e);
    const t = e => new Promise((t => setTimeout(t, e)));
    window.addEventListener("message", (async function (e) {
        if (!e.data.url) {
            return;
        }
        const o = (await chrome.storage.local.get("settings")).settings;
        if (o) {
            if (e.data.url.includes("/api/v3/lesson/presentation/fetch") && o.autoAnswer) {
                if (e.data instanceof Object) {
                    console.log("收到雨课堂课件信息: ", e.data.url, e.data);
                    const t = JSON.parse(e.data.data).data.slides.filter((e => Object.keys(e).includes("problem"))).map((e => e.problem));
                    chrome.storage.local.set({
                        problems: t
                    }),
                        console.log("找到雨课堂问题信息: ", t),
                        function () {
                            const e = document.createElement("button");
                            e.style.position = "fixed",
                                e.style.top = "10px",
                                e.style.left = "10px",
                                e.style.background = "#639ef4",
                                e.style.height = "30px",
                                e.style.color = "white",
                                e.style.padding = "0 10px",
                                e.textContent = "点击此处允许播放提醒音",
                                e.onclick = () => {
                                    chrome.storage.local.set({
                                        notificationSound: !0
                                    }),
                                        e.remove()
                                }
                                ,
                                document.children[0]?.appendChild(e),
                                console.log("初始化按钮完成")
                        }
                }
            }
        }
    }
    ))

    if (location.href.match(/yuketang.cn/)) {
        Rainclassroom();
        console.log(1)
    }
    function JumpObj(elem, range, startFunc, endFunc) {
        // 按钮跳动
        var curMax = range = range || 6;
        startFunc = startFunc || function () { };
        endFunc = endFunc || function () { };
        var drct = 0;
        var step = 1;

        init();

        function init() { elem.style.position = 'relative'; active() }
        function active() { elem.onmouseover = function (e) { if (!drct) jump() } }
        function deactive() { elem.onmouseover = null }

        function jump() {
            var t = parseInt(elem.style.top);
            if (!drct) motionStart();
            else {
                var nextTop = t - step * drct;
                if (nextTop >= -curMax && nextTop <= 0) elem.style.top = nextTop + 'px';
                else if (nextTop < -curMax) drct = -1;
                else {
                    var nextMax = curMax / 2;
                    if (nextMax < 1) { motionOver(); return; }
                    curMax = nextMax;
                    drct = 1;
                }
            }
            setTimeout(function () { jump() }, 200 / (curMax + 3) + drct * 3);
        }
        function motionStart() {
            startFunc.apply(this);
            elem.style.top = '0';
            drct = 1;
        }
        function motionOver() {
            endFunc.apply(this);
            curMax = range;
            drct = 0;
            elem.style.top = '0';
        }

        this.jump = jump;
        this.active = active;
        this.deactive = deactive;
    };

    function loadImage(url) {
        // 图片加载
        return new Promise((resolve, reject) => {
            var img = new Image();
            var data;
            img.setAttribute("crossOrigin", "Anonymous");
            img.src = url;
            img.onError = function () {
                throw new Error('Cannot load image: "' + url + '"');
            };
            img.onload = function () {
                var canvas = document.createElement("canvas");
                document.body.appendChild(canvas);
                canvas.width = img.width;
                canvas.height = img.height;
                var ctx = canvas.getContext("2d");
                ctx.drawImage(img, 0, 0);
                // Grab the image as a jpeg encoded in base64, but only the data
                data = canvas
                    .toDataURL("image/jpeg")
                    .slice("data:image/jpeg;base64,".length);
                // Convert the data to binary form
                data = atob(data);

                document.body.removeChild(canvas);
                resolve(data);

            };
        });
    }
    function txtdownload(result) {
        // txt下载
        const stringData = JSON.stringify(result, null, 2)
        // dada 表示要转换的字符串数据，type 表示要转换的数据格式
        const blob = new Blob([stringData], {
            type: 'application/json'
        })
        // 根据 blob生成 url链接
        const objectURL = URL.createObjectURL(blob)

        // 创建一个 a 标签Tag
        const aTag = document.createElement('a')
        // 设置文件的下载地址
        aTag.href = objectURL
        // 设置保存后的文件名称
        aTag.download = "1.txt"
        // 给 a 标签添加点击事件
        aTag.click()
        // 释放一个之前已经存在的、通过调用 URL.createObjectURL() 创建的 URL 对象。
        // 当你结束使用某个 URL 对象之后，应该通过调用这个方法来让浏览器知道不用在内存中继续保留对这个文件的引用了。
        URL.revokeObjectURL(objectURL)
    }

    function getImgWidthHeight(src) {
        // 获取图片宽高
        return new Promise((resolve, reject) => {
            const img = new Image()
            img.src = src
            // 图片是否有缓存 如果有缓存可以直接拿 如果没有缓存 需要从onload拿
            if (img.complete) {
                const { width, height } = img
                resolve({
                    width,
                    height,
                })
            } else {
                img.onload = function () {
                    const { width, height } = img
                    resolve({
                        width,
                        height,
                    })
                }
            }
        })
    }

    async function download(imgList, namepdf) {
        // 进度条建立
        var table3 = document.createElement('div');
        document.getElementById("app").append(table3);
        table3.className = 'wrapper';
        table3.id = 'bar';
        table3.style.zIndex = 19891015;
        table3.style.width = '60%';
        table3.style.height = '5%';
        table3.style.position = 'absolute';
        table3.style.top = '30%';
        table3.style.left = '20%';
        table3.style.visibility = 'visible';
        table3.style.background = 'rgba(255,235,205,0.5)';
        table3.style.filter = 'alpha(opacity=50)';
        table3.style.borderRadius = '20px 20px 20px 20px';
        table3.style.textAlign = 'center';
        table3.style.overflowY = null;

        var sho = document.createElement('div');
        table3.append(sho)
        sho.style.width = '100%';
        sho.style.height = '100%';
        sho.style.top = '50%';
        sho.style.alignItems = 'center';
        sho.style.textAlign = 'center';


        var table4 = document.createElement('div');
        table4.id = 'jd';
        table4.className = 'first';
        table3.append(table4);
        table4.style.zIndex = 19891015;

        table4.style.height = '80%';
        table4.style.position = 'absolute';
        table4.style.top = '10%';
        table4.style.left = '10%';
        table4.style.visibility = 'visible';
        table4.style.background = 'rgba(112, 128, 144, 0.5)';
        table4.style.filter = 'alpha(opacity=50)';
        table4.style.borderRadius = '20px 20px 20px 20px';
        table4.style.textAlign = 'center';
        table4.style.overflowY = null;


        // pdf下载
        var imgData = new Array();
        console.log(imgList)
        for (var i = 0; i < imgList.length; i++) {
            var link = imgList[i];
            console.log("获取第" + i + "张图片");
            table4.style.width = String((i + 1) / imgList.length * 80) + '%';
            sho.innerHTML = 'PDF下载中：第' + String(i + 1) + '页，共' + String(imgList.length) + '页';
            await loadImage(link).then((data) => {
                imgData.push(data);
            });
        }

        const { width, height } = await getImgWidthHeight(imgList[0])

        var doc = new jsPDF({
            orientation: "landscape",
            unit: "px",
            format: [width, height],
        });

        const output = namepdf + ".pdf";
        let idx = 0;
        imgData.forEach((e) => {
            idx++;
            doc.addImage(e, "JPG", 0, 0, width, height);
            if (idx < imgData.length) {
                doc.addPage();
            }
        });
        sho.innerHTML = '下载完成';
        doc.save(output);
        sho.remove();
        table4.remove();
        table3.remove();
    }

    function Rainclassroom() {
        // 新建答案填充框
        var table1 = document.createElement('div');
        table1.className = 'answer';
        document.getElementById("app").append(table1);
        table1.style.zIndex = 19891015;
        table1.style.width = '20%';
        //table1.style.height = '11%';
        table1.style.position = 'absolute';
        table1.style.top = '60%';
        table1.style.left = '70%';
        table1.style.visibility = 'visible';
        table1.style.background = 'rgba(255,235,205,0.5)';
        table1.style.filter = 'alpha(opacity=50)';
        table1.style.borderRadius = '20px 20px 20px 20px';
        table1.style.textAlign = 'center';
        table1.style.overflowY = null;

        // 刷课
        var tr4 = document.createElement("button");
        table1.appendChild(tr4)
        Object.assign
            (
                tr4.style,
                {
                    width: '40%',
                    borderradius: '8px',
                    background: 'rgba(255, 251, 240, 0.3)',
                    margin: '5%',
                    alignitems: 'center',
                    justifycontent: 'center',
                    textAlign: 'center',
                    filter: 'alpha(opacity=50%)'
                }
            )
        tr4.innerText = "开始刷课";
        const button2 = tr4;
        tr4.className = 'close';



        function showLogin() {
            // 连续翻页设置
            document.onkeydown = function (ev) {
                var event = ev || event
            };
            var e = new KeyboardEvent('keydown', { 'keyCode': 40, 'which': 40 });
            document.dispatchEvent(e);
        }

        button2.onclick = () => {
            // 连续翻页
            var check = tr4.className;
            if (check == "close") {
                tr4.className = 'open';
                tr4.innerText = '结束刷课';
                //setInterval方法或字符串 ，毫秒，参数数组（方法的）)
                if (!window.interVal) {
                    window.interVal = window.setInterval(() => {
                        showLogin();
                    }, (1.5 * 1000 + Math.random() * 1.5 * 1000));
                }
            }
            if (check == "open") {
                tr4.className = 'close';
                tr4.innerText = '开始刷课';
                if (window.interVal) {
                    window.clearInterval(window.interVal);
                    window.interVal = null;
                }
            }
        }

        var tr1 = document.createElement("button");
        table1.appendChild(tr1)
        Object.assign
            (
                tr1.style,
                {
                    width: '40%',
                    borderradius: '8px',
                    background: 'rgba(255, 251, 240, 0.3)',
                    margin: '5%',
                    alignitems: 'center',
                    justifycontent: 'center',
                    textAlign: 'center',
                    filter: 'alpha(opacity=50%)'
                }
            )
        tr1.innerText = "显示答案";
        const button0 = tr1;
        tr1.className = 'close';

        button0.onclick = () => {
            //console.log(tr1.className)
            var check = tr1.className;
            if (check == "close") {
                tr1.className = 'open';
                tr1.innerText = '隐藏答案';

                Object.assign
                    (
                        table1.style,
                        {
                            height: '20%',
                            overflowY: 'overlay'
                        }
                    )

                var lists = document.getElementsByClassName('slide');
                for (var j = 0; j < lists.length; j++) {
                    var tri = lists[j];
                    Object.assign
                        (
                            tri.style,
                            {
                                display: null
                            }
                        )

                }
            }
            if (check == "open") {
                tr1.className = 'close';
                tr1.innerText = '显示答案';
                Object.assign
                    (
                        table1.style,
                        {
                            height: '5%',
                            overflowY: null
                        }
                    )
                var lists = document.getElementsByClassName('slide');
                for (var j = 0; j < lists.length; j++) {
                    var tri = lists[j];
                    Object.assign
                        (
                            tri.style,
                            {
                                display: 'none',
                            }
                        )
                }
            }
        }
        //JumpObj(button0, 10)


        var ppti = 0; // ppt个数
        const Allanswer = {} // 所有答案
        


        //监听操作事件
        const handleSendResule = (res) => {
            const result = JSON.parse(res.target.responseText); // 返回值
            if (result['data'] == undefined) // 判断是否为目标xhr
            {
                return;
            }
            if (result.data['slides'] == undefined && result.data['Slides'] == undefined) {
                return;
            }
            if (result.data.Slides) {
                var namei = result.data.Title;
                var checkpdf = document.getElementById(namei);

                if (checkpdf == null) {
                    ppti += 1;
                    var tr3 = document.createElement("div");
                    tr3.id = result.data.Title;
                    tr3.innerText = "PPT：" + result.data.Title + ".pdf" + "不支持解析";
                    tr3.className = 'slide';
                    if (tr1.className == 'close') {
                        tr3.setAttribute("style", "display:none")
                    }
                    if
                        (tr1.className == 'open') {
                        tr3.setAttribute("style", "display:block")
                    }
                    table1.appendChild(tr3)
                }



            }
            else {
                if (result.data.slides) {
                    console.log("czy打印的results" + result.data);
                    var namepdf = '';
                    var answer = result.data.slides;
                    const imgList = Array();
                    if (typeof (answer) == "object") {
                        var lens = answer.length;
                        var key = Object.keys(answer);
                        var i = 0;
                        var temp3 = '';
                        var nullpage = 0;
                        for (i = 0; i < lens; i++) {
                            imgList[i] = result.data.slides[i].cover;
                            if (imgList[i] == '') {
                                nullpage = nullpage + 1;
                            }
                            if (result.data.presentation) // ppt回看时的答案
                            {

                                namepdf = result.data.presentation.title;
                                var que1 = result.data.slides[i].answer;
                                var a, kk;
                                if (que1 != null) {
                                    a = que1.correctAnswer;
                                    for (kk = 0; kk < a.length; kk++) {
                                        temp3 = temp3 + a[kk] + ";"
                                    }
                                    Allanswer[key[i]] = temp3
                                }
                                que1 = result.data.slides[i].problem;
                                if (que1 != null) {
                                    a = que1.content.answer;
                                    for (kk = 0; kk < a.length; kk++) {
                                        temp3 = temp3 + a[kk] + ";"
                                    }
                                    Allanswer[key[i]] = temp3

                                }
                                console.log(que1)

                                temp3 = ""
                            }
                            else {
                                // 上课的
                                namepdf = result.data.title;
                                var que = answer[key[i]].problem;
                                if (typeof (t) != "undefined" && typeof (t) != "null") {
                                    len2 = t[i].result.length
                                    //console.log(len2)
                                    if (len2 == 0) {
                                        Allanswer[key[i]] = "答案任意"
                                    }
                                    else {
                                        for (var j = 0, len2; j < len2; j++) {
                                            temp3 = temp3 + t[i].result + "; "
                                            //console.log(temp2)
                                        }
                                        Allanswer[key[i]] = temp3

                                    }
                                    temp3 = ""
                                }
                            }
                        }
                    }
                    var checkpdf = document.getElementById(namepdf);
                    // 检查是否存在
                    if (checkpdf == null) {
                        ppti += 1;
                        var tr3 = document.createElement("div");
                        tr3.innerText = "PPT：" + namepdf + "的答案为：";
                        tr3.id = namepdf;
                        tr3.className = 'slide';
                        if (tr1.className == 'close') {
                            tr3.setAttribute("style", "display:none")
                        }
                        if
                            (tr1.className == 'open') {
                            tr3.setAttribute("style", "display:block")
                        }
                        table1.appendChild(tr3)

                        // 答案加入
                        var ind = Object.keys(Allanswer)
                        var ans = Object.values(Allanswer)
                        for (i = 0; i < ind.length; i++) {
                            var tr2 = document.createElement("div");
                            tr2.className = 'slide';
                            if (tr1.className == 'close') {
                                tr2.setAttribute("style", "display:none")
                            }
                            if
                                (tr1.className == 'open') {
                                tr2.setAttribute("style", "display:block")
                            }
                            //tr2.setAttribute("style", "background:rgba(85,85,85,0.7);color:white;background:rgba(85,85,85,0.7);color:white;")
                            tr2.innerText = String(parseInt(ind[i]) + 1) + "题答案为：\t" + ans[i];
                            table1.appendChild(tr2)
                        }
                        // 文件写入按钮
                        var button1 = document.createElement('button');

                        table1.appendChild(button1);
                        if (tr1.className == 'close') {
                            button1.setAttribute("style", "display:none")
                        }
                        if
                            (tr1.className == 'open') {
                            button1.setAttribute("style", "display:null")
                        }
                        Object.assign
                            (
                                button1.style,
                                {
                                    width: '50%',
                                    borderradius: '8px',
                                    background: 'rgba(255, 251, 240, 0.3)',
                                    alignitems: 'center',
                                    justifycontent: 'center',
                                    textAlign: 'center',
                                    filter: 'alpha(opacity=50%)'
                                }
                            )
                        button1.className = 'slide';
                        if (nullpage > 0) {
                            button1.innerText = '保存为PDF(加载ing)';
                        }
                        else {
                            button1.innerText = '保存为PDF(finished)';
                        }
                        button1.id = 'down' + namepdf;
                        const button = button1;
                        // 给按钮添加点击事件
                        button.onclick = () => {
                            // 要保存的字符串, 需要先将数据转成字符串
                            // txtdownload();
                            download(imgList, namepdf);
                        }
                        // 跳动效果
                        JumpObj(button, 10)
                    }
                    else {
                        if (nullpage > 0) {
                            button1.innerText = '保存为PDF(加载ing)';
                        }
                        else {
                            button1.innerText = '保存为PDF(finished)';
                        }
                        button1 = document.getElementById('down' + namepdf);
                        button1.onclick = () => {
                            // 要保存的字符串, 需要先将数据转成字符串
                            // txtdownload();
                            download(imgList, namepdf);
                        }
                    }
                }
            }
        }
        const handler = {
            apply: function (target, thisArg, body) {
                thisArg.addEventListener('load', handleSendResule)
                thisArg.addEventListener('error', handleSendResule)
                return target.call(thisArg, body)
            }
        }
        XMLHttpRequest.prototype.send = new Proxy(XMLHttpRequest.prototype.send, handler)
    }

    // Your code here...
})();
