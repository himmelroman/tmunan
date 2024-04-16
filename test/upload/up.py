import requests

url = 'http://127.0.0.1:8080/api/imagine/img2img_upload'
# files = [('files', open('images/1.png', 'rb')), ('files', open('images/2.png', 'rb'))]
files = [('file', open('/Users/himmelroman/Desktop/Bialik/snippet/out.png', 'rb'))]
resp = requests.post(url=url, files=files)
print(resp.json())
