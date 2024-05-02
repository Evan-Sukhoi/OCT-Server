from http.server import BaseHTTPRequestHandler, HTTPServer
import os
import io
import random
import string
from PIL import Image
import cgi
from PIL import ImageEnhance
import time

import torch
from PIL import Image
import torchvision.transforms as transforms
from matplotlib import pyplot as plt

from iMED.build.CTTIF import CTTIF
from Regan import inference_realesrgan

UPLOAD_DIR = 'uploads'
PROCESSED_DIR = 'processed'
NUM_PROCESSED_IMAGES = 128

model_1 = CTTIF('iMED/config/test.yaml')
model_2_path = 'Regan/models/net_g_1000000.pth'

stage_1_output_path = 'iMED/test_images/modetest'

def stage_1(input_path_cf):
    print(input_path_cf)
    image_transform = transforms.Compose([
        transforms.Resize((128, 128)),  
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    image1 = Image.open(input_path_cf)
    image1 = image_transform(image1)
    image1 = image1.unsqueeze(0)  

    model_1.test(image1)
    
def stage_2(output_path):
    inference_realesrgan.deblur(model_2_path, stage_1_output_path, output_path)
    

class MyServer(BaseHTTPRequestHandler):
    name = ''
    task_path = ''
    cf_file_name = ''
    oct_file_name = ''
    def do_POST(self):
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={'REQUEST_METHOD': 'POST'}
        )
        if 'name' in form:
            name = form['name'].value
            MyServer.task_path = os.path.join(UPLOAD_DIR, name)
            MyServer.cf_file_name = 'cf-' + name + '.png'
            MyServer.oct_file_name = 'oct-' + name + '.png'
            while True:
                try:
                    os.mkdir(MyServer.task_path)
                    break
                except FileExistsError:
                    MyServer.task_path = os.path.join(UPLOAD_DIR, ''.join(random.choices(string.ascii_uppercase + string.digits, k=10)))
            self.send_response(200)
            self.end_headers()
            print(f"Received image pair: {name}")
            os.mkdir(os.path.join(MyServer.task_path, 'processed'))
        elif 'file' in form:
            file_item = form['file']
            original_filename = file_item.filename
            
            if self.path.endswith('/uploadCF'):
                with open(os.path.join(MyServer.task_path, MyServer.cf_file_name), 'wb') as f:
                    f.write(file_item.file.read())
            elif self.path.endswith('/uploadOCT'):
                file = file_item.file.read()
                img = Image.open(io.BytesIO(file))
                img = img.crop((0, 30, 512, 542))
                with open(os.path.join(MyServer.task_path, MyServer.oct_file_name), 'wb') as f:
                    img.save(f, 'PNG')

            self.send_response(200)
            self.end_headers()
            self.wfile.write(f"Uploaded: {original_filename}".encode())
        else:
            self.send_response(400)
            self.end_headers()
            self.wfile.write("No file uploaded".encode())

    def do_GET(self):
        if self.path == '/generateImages':
            print("Generating images...")
            self.process_image(os.path.join(MyServer.task_path, MyServer.cf_file_name), \
                                os.path.join(MyServer.task_path, MyServer.oct_file_name), \
                                os.path.join(MyServer.task_path, 'processed'))
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            for file_name in os.listdir(os.path.join(MyServer.task_path, 'processed')):
                with open(os.path.join(MyServer.task_path, 'processed', file_name), 'rb') as f:
                    self.wfile.write(f.read())
        elif self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            with open('index.html', 'rb') as f:
                self.wfile.write(f.read())
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write("Not found".encode())
            

    def process_image(self, input_cf, input_oct, output_path):
        stage_1(input_cf)
        stage_2(output_path)

def run(server_class=HTTPServer, handler_class=MyServer, port=8080):
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f"Server running on port {port}")
    httpd.serve_forever()

if __name__ == '__main__':
    run()
