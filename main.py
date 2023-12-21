import asyncio
import cv2
import imutils
import numpy as np
import requests
import io
import os

from PIL import Image, ImageFilter
from imutils.contours import sort_contours
from aiogram import Bot, Dispatcher, types, F, Router
from aiogram.types import Message
from aiogram.filters import Command
from aiogram.enums.parse_mode import ParseMode
from aiogram.fsm.storage.memory import MemoryStorage
from tensorflow import keras

import config

# creating bot object
bot = Bot(token=config.BOT_TOKEN, parse_mode=ParseMode.HTML)
router = Router()
model = keras.models.load_model('model.h5')

# link to file on telegram servers
URI_INFO = f'https://api.telegram.org/bot{config.BOT_TOKEN}/getFile?file_id='
# link to image itself
URI = f'https://api.telegram.org/file/bot{config.BOT_TOKEN}/'

async def main():
    print('running')
    # MemoryStorage - all usaved bot data will be deleted after session
    dp=Dispatcher(storage=MemoryStorage())
    dp.include_router(router)
    # deletes all updates after last bot's shutdown
    await bot.delete_webhook(drop_pending_updates=True)
    # starting bot
    await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())

@router.message(Command('start'))
async def start_handler(msg: Message):
    await msg.answer('Hello! This bot can solve simple mathematic examples. Just write it on paper and send here!')

@router.message()
async def get_photo(msg: types.Message):
    # getting file id
    file_id = msg.photo[-1].file_id
    # getting image by that id on telegram servers
    resp = requests.get(URI_INFO + file_id)
    # getting path to that image on servers
    img_path = resp.json()['result']['file_path']
    # downloading it
    img = requests.get(URI+img_path)
    # openning image and converting it so we could read it
    img = Image.open(io.BytesIO(img.content))
    # making directory for images and saving image
    if not os.path.exists('static'):
       os.mkdir('static')
    img.save('static/image.png', format='PNG')
    # loading image to model
    try:
        solution = test_pipeline('static/image.png')
        await msg.answer(f'Solution to your example {solution[0]} is {solution[1]}')
    except:
       await msg.answer(f'An error occured while detecting symbols, try another image')
    # deleting image
    os.remove('static/image.png')

def test_pipeline(image_path):
    # a list for detected symbols
    chars = []
    img = cv2.imread(image_path)
    img = cv2.resize(img, (800, 800))
    # turn image to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # drawing edges of symbols on images
    edged = cv2.Canny(img_gray, 30, 150)
    # getting edges and sorting them left to right to read expression
    contours = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sort_contours(contours, method="left-to-right")[0]
    labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'add', 'div', 'dec', 'mul', 'sub']

    for c in contours:
        # here we're getting height and width of the box of a symbol
        (x, y, w, h) = cv2.boundingRect(c)
        # this is such a crutch but I couldn't come up with better solution
        if 20<=w:
            # getting a region of interest for a symbol
            roi = img_gray[y:y+h, x:x+w]
            # setting threshold to make an image b&w
            thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            # resizing the image so that both height and width were less than
            # 32 pixels so we could put an image to predictor
            (th, tw) = thresh.shape
            if tw > th:
                thresh = imutils.resize(thresh, width=32)
            if th > tw:
                thresh = imutils.resize(thresh, height=32)
            (th, tw) = thresh.shape
            dx = int(max(0, 32 - tw) / 2.0)
            dy = int(max(0, 32 - th) / 2.0)
            # adding padding to image to make it square-shaped (32x32)
            padded = cv2.copyMakeBorder(thresh,top=dy,bottom=dy,left=dx,right=dx,borderType=cv2.BORDER_CONSTANT,value=(0, 0, 0))
            padded = cv2.resize(padded, (32, 32))
            padded = np.array(padded)
            # binarize an image
            padded = padded/255.
            # reshaping image to (batch_size, height, width)
            padded = np.expand_dims(padded, axis=0)
            # reshaping image to (batch_size, height, width, num_channels)
            padded = np.expand_dims(padded, axis=-1)
            # so now our image is 32x32 and and it's binarized
            # and we can predict it's label
            pred = model.predict(padded)
            pred = np.argmax(pred, axis=1)
            label = labels[pred[0]]
            chars.append(label)
            # drawing bounding boxes and predicted label over image
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(img, label, (x-5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

    # changing labels to operators and calculating result
    expression = ''

    for char in chars:
      if char == 'add':
        char = '+'
      elif char == 'sub':
        char = '-'
      elif char == 'div':
        char = '/'
      elif char == 'mul':
        char = '*'

      expression += char

    answer = [expression, eval(expression)]

    return answer

if __name__ == '__main__':
    asyncio.run(main())