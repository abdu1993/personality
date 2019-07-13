from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
import uvicorn, aiohttp, asyncio
from io import BytesIO
import sys
from pathlib import Path
import csv
import time

from fastai import *
from fastai.text import *

model_file_url = 'https://drive.google.com/uc?export=download&id=1xjqUCB912sMTGp3Y6NTwoPODsM1pdPGc'
model_file_name = 'poetmodel'
path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))

async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f: f.write(data)

async def setup_learner():
    await download_file(model_file_url, path/'models'/f'{model_file_name}.pth')
    data_lm = TextLMDataBunch.load(path/'static', 'data_lm')
    data_bunch = (TextList.from_csv(path, csv_name='static/blank.csv', vocab=data_lm.vocab)
        .random_split_by_pct()
        .label_for_lm()
        .databunch(bs=10))
    learn = language_model_learner(data_bunch, pretrained_model=None)
    learn.load(model_file_name)
    return learn

loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()

@app.route('/')
def index(request):
    html = path/'view'/'index.html'
    return HTMLResponse(html.open().read())

@app.route('/analyze', methods=['POST'])
async def analyze(request):
    data = await request.form()

    return JSONResponse({'result': textResponse(data)})

def textResponse(data):
    
    t = t = {'INTJ':'INTJ - Architect - Imaginative and stratigic thinkers with a plan for everything.',
    'INTP':'INTP - Logician - Innovative inventors with an unquenchable thirst for knowledge.',
    'ENTJ':'ENTJ - Commander - Bold, imaginative, and strong willed leaders, always finding a way - or making one.',
    'ENTP':'ENTP - Debater - Smart and curious thinkers who cannot resist an intellectual challenge.',
    'INFJ':'INFJ - Advocate - Quiet and mystical, yet very inspiring and tireless idealists.',
    'INFP':'INFP - Mediator - Poetic, kind, and altruistic people, always eager to help a good cause.',
    'ENFJ':'ENFJ - Protagonist - Charismatic and inspiring leaders, able to mesmerize their listeners.',
    'ENFP':'ENFP - Campaigner - Enthusiastic, creative, and sociable free spirits, who can always find a reason to smile.',
    'ISTJ':'ISTJ - Logistician - Practical and fact-minded individuals, whose reliability cannot be doubted.',
    'ISFJ':'ISFJ - Defender - Very dedicated and warm protectors, always ready to defend their loved ones.',
    'ESTJ':'ESTJ - Executive - Excellent administrators, unsurpassed at managing things - or people.',
    'ESFJ':'ESFJ - Consul - Extraordinarily caring, social, and popular people, always eager to help.',
    'ISTP':'ISTP - Virtuoso - Bold and practical experimenters, masters of all kinds of tools.',
    'ISFP':'ISFP - Adventurer - Flexible and charming artists, always ready to explore and experience something new.',
    'ESTP':'ESTP - Entrepreneur - Smart, energetic, and very perceptive people, who truly enjoy living on the edge.',
    'ESFP':'ESFP - Entertainer - Spontaneous, energetic, and enthusiastic people - life is never boring around them.'}
    
    pred = learn.predict(data['file'])
    time.sleep(2)
    pred = str(pred[0])
    return t.get(pred, 0)

if __name__ == '__main__':
    if 'serve' in sys.argv: uvicorn.run(app, host='0.0.0.0', port=5042)
