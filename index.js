const img = document.querySelector('.img')
const submit = document.querySelector('#submit')
const debutton_1 = document.querySelector('#delete_1')
debutton_1.style.display = 'none'
debutton_1.addEventListener('click', function () {

    const upimg_1 = document.querySelector('#upimg_1')
    // const hr = document.querySelector('hr')
    if (upimg_1 === null) return
    img.removeChild(upimg_1)
    // img.removeChild(hr)
    img_1.style.display = ''
    debutton_1.style.display = 'none'
    submit.style.backgroundColor = '#9ac6fd'
    cf_file = ''
})

// const debutton_2 = document.querySelector('#delete_2')
// debutton_2.style.display = 'none'
// debutton_2.addEventListener('click', function () {
//     const upimg_2 = document.querySelector('#upimg_2')
//     if (upimg_2 === null) return
//     img.removeChild(upimg_2)
//     img_2.style.display = ''
//     debutton_2.style.display = 'none'
//     submit.style.backgroundColor = '#9ac6fd'
//     oct_file = ''
// })

const pickerOpts = {
    types: [
        {
            description: "Images",
            accept: {
                "image/*": [".png", ".gif", ".jpeg", ".jpg"],
            },
        },
    ],
    excludeAcceptAllOption: true,
    multiple: false,
}

const img_1 = document.querySelector('#img_1')
let cf_file;
img_1.addEventListener('click', function () {
    window.showOpenFilePicker(pickerOpts).then(function ([fileHandle]) {
        fileHandle.getFile().then(function (file) {
            cf_file = file;
            let src = URL.createObjectURL(file)
            img_1.style.display = 'none'
            img.insertAdjacentHTML('afterbegin', `<img id="upimg_1" src="${src}">
            `)
            debutton_1.style.display = ''
            // const upimg_2 = document.querySelector('#upimg_2')
            // console.log(upimg_2)
            // if (upimg_2 !== null) {
            submit.style.backgroundColor = '#4096ff'
            // }
        }).catch(function (error) {
            console.error('Error getting file:', error);
        });
    }).catch(function (error) {
        console.error('Error selecting file:', error);
    });
});
// const img_2 = document.querySelector('#img_2')
// let oct_file;
// img_2.addEventListener('click', function () {
//     window.showOpenFilePicker(pickerOpts).then(function ([fileHandle]) {
//         fileHandle.getFile().then(function (file) {
//             oct_file = file;
//             let src = URL.createObjectURL(file)
//             img_2.style.display = 'none'
//             img.insertAdjacentHTML('beforeend', `<img id="upimg_2" src="${src}">`)
//             debutton_2.style.display = ''
//             const upimg_1 = document.querySelector('#upimg_1')
//             if (upimg_1 !== null) {
//                 submit.style.backgroundColor = '#4096ff'
//             }
//         }).catch(function (error) {
//             console.error('Error getting file:', error);
//         });
//     }).catch(function (error) {
//         console.error('Error selecting file:', error);
//     });
// });


let imgs = [];
submit.addEventListener('click', function () {
    const upimg_1 = document.querySelector('#upimg_1')
    if (upimg_1 !== null) {
        if (cf_file !== null) {
            const small_window = document.querySelector('.switch')
            small_window.innerHTML = ''
            const disimg = document.querySelector('#disimg')
            disimg.src = '#'
            let imgs = []

            var cfFileName = '';
            var octFileName = '';
            var taskName = new FormData();


            var cfFormData = new FormData();
            // var cfFileInput = document.getElementById('cfImage');

            // cfFormData.append('file', cfFileInput.files[0]);
            cfFormData.append('file', cf_file);

            // cfFileName = cfFileInput.files[0].name;
            cfFileName = cf_file.name;

            taskName.append('name', cfFileName.split('.').slice(0, -1).join('.'));

            var nameXhr = new XMLHttpRequest();
            var cfXhr = new XMLHttpRequest();


            nameXhr.open('POST', 'http://localhost:8080/uploadName', true);
            cfXhr.open('POST', 'http://localhost:8080/uploadCF', true);
            cfXhr.onload = function () {
                if (cfXhr.status == 200) {
                    alert('成功上传CF图片');
                } else {
                    alert('上传CF图片失败');
                }
            };

            nameXhr.onload = function () {
                if (nameXhr.status == 200) {
                    cfXhr.send(cfFormData);
                } else {
                    alert('上传名称失败');
                }
            };

            nameXhr.send(taskName);
            taskName.delete('name');

            setTimeout(() => {


                // xhr.responseType = 'arraybuffer'; // 设置响应类型为数组缓冲区
                for (let i = 0; i < 128; i++) {
                    let img = { "src": `./assets/img/loading.png`, "id": `${i}` }
                    imgs.push(img)

                }
                const small_window = document.querySelector('.switch')
                const hr = document.querySelector('hr')

                const nums = imgs.length
                const disimg = document.querySelector('#disimg')
                let red_pico;
                const node = document.createElement('div')
                node.className = 'pico'
                node.style.backgroundColor = 'red'
                node.id = `pico${0}`
                node.dataPath = imgs[0].src
                node.dataId = imgs[0].id
                node.innerHTML = `<img src="${imgs[0].src}" class="smpc" id="img0"> ${imgs[0].id}`
                small_window.appendChild(node)
                red_pico = node
                disimg.src = node.dataPath
                const pico_width = red_pico.getBoundingClientRect().width
                for (let i = 1; i < imgs.length; i++) {
                    const node = document.createElement('div')
                    node.className = 'pico'
                    node.id = `pico${i}`
                    node.innerHTML = `<img src="${imgs[i].src}" class="smpc" id="img${i}"> ${imgs[i].id}`
                    node.dataPath = imgs[i].src
                    node.dataId = imgs[i].id
                    small_window.appendChild(node)
                }
                const picos = document.querySelectorAll('.pico')
                for (const pico of picos) {
                    pico.addEventListener('click', function (e) {
                        const click_pico = e.currentTarget
                        red_pico.style.backgroundColor = ''
                        click_pico.style.backgroundColor = 'red'
                        red_pico = click_pico
                        disimg.src = click_pico.dataPath
                        hr.style.right = (100 - 100 / nums * click_pico.dataId) + '%'
                    })
                }


                const left_swicth = document.querySelector('.bi-caret-left')
                left_swicth.addEventListener('click', function () {
                    const previous_pico = red_pico.previousSibling

                    if (previous_pico.nodeName !== '#text') {
                        small_window.scrollLeft = small_window.scrollLeft - pico_width

                        red_pico.style.backgroundColor = ''
                        previous_pico.style.backgroundColor = 'red'
                        red_pico = previous_pico
                        disimg.src = previous_pico.dataPath
                        hr.style.right = (100 - 100 / nums * red_pico.dataId) + '%'

                    }

                })
                const right_swicth = document.querySelector('.bi-caret-right')
                right_swicth.addEventListener('click', function () {
                    const next_pico = red_pico.nextSibling

                    if (next_pico.nodeName !== '#text') {
                        small_window.scrollLeft = small_window.scrollLeft + pico_width

                        red_pico.style.backgroundColor = ''
                        next_pico.style.backgroundColor = 'red'
                        red_pico = next_pico
                        disimg.src = next_pico.dataPath
                        hr.style.right = (100 - 100 / nums * red_pico.dataId) + '%'
                    }
                })

                var xhr = new XMLHttpRequest();

                xhr.open('GET', 'http://localhost:8080/generateImages', true);

                let directoryPath;
                xhr.onload = function () {
                    if (xhr.status == 200) {
                        directoryPath = xhr.response;
                    } else {
                        alert('Error generating images');
                    }
                };

                xhr.send();


                setTimeout(() => {
                    directoryPath = directoryPath.replace(/\\/g, "/");
                    function getFileStatus(src) {
                        return new Promise((resolve, reject) => {
                            xhr.open('GET', `http://localhost:8080/getFile?path=${src}`, true);
                            xhr.onload = function () {
                                if (xhr.status == 200) {
                                    resolve(); // 文件状态为200，执行 resolve
                                } else {
                                    reject(); // 文件状态不为200，执行 reject
                                }
                            };
                            xhr.send();
                        });
                    }

                    async function checkFileStatusAndContinue(i) {
                        console.log(`${i}`);
                        let src = `${directoryPath}/imgoutput_no${i}_out.png`;

                        try {
                            await getFileStatus(src); // 等待文件状态获取
                            if (i == 0) {
                                const disimg = document.querySelector("#disimg")
                                disimg.src = "getFile?path=" + src
                            }
                            imgs[i].src = "getFile?path=" + src;
                            const img = document.querySelector(`#img${i}`)
                            img.src = "getFile?path=" + src;
                            const node = document.querySelector(`#pico${i}`)
                            node.dataPath = "getFile?path=" + src;

                            console.log("File loaded successfully");
                            if (i < 127) { // 继续下一次查询
                                checkFileStatusAndContinue(i + 1);
                            } else {
                                alert("所有的OCT已生成")
                            }
                        } catch (error) {
                            console.log("202");
                            setTimeout(() => {
                                checkFileStatusAndContinue(i); // 等待一段时间后重新查询
                            }, 1000);
                        }
                    }


                    checkFileStatusAndContinue(0)
                }, 1000)


                // xhr.send();
            }, 1000)
        } else {
            window.alert("请选择CF图片")
        }

    } else {
        window.alert("请选择CF图片")
    }
})







