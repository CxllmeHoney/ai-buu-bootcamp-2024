import requests
from io import BytesIO
from PIL import Image
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from linebot.v3 import WebhookHandler
from linebot.v3.messaging import (Configuration,
                                  ApiClient,
                                  MessagingApi,
                                  ReplyMessageRequest,
                                  TextMessage)
from linebot.v3.webhooks import (MessageEvent,
                                 TextMessageContent,
                                 ImageMessageContent)
from linebot.v3.exceptions import InvalidSignatureError
import google.generativeai as genai

app = FastAPI()

# ข้อมูล token และ channel secret สำหรับ LINE
ACCESS_TOKEN = "29DtEHVbIAioorYrV9cpGqpB/mR0eyJ/Sn3M9kHHX81xZ27dvyeA0YXlyDtTtIIBNJ5cFhiJAWpDUw8C63caD/LcTybuzeYMgHqkyhWrl0kZZrHWNiazi3/0zluFQRwZSzAozVG4Iyd8zFLkTVvoKQdB04t89/1O/w1cDnyilFU="
CHANNEL_SECRET = "270eef1263d1e55cf8f8d4abf041e3e9"

# ข้อมูล Gemini api key
GEMINI_API_KEY = "AIzaSyCZruMeHjowdcGne-dpfCfxJ4lVk5pnrPQ"

# การเชื่อมต่อ และตั้งค่าข้อมูลเพื่อเรียกใช้งาน LINE Messaging API
configuration = Configuration(access_token=ACCESS_TOKEN)
handler = WebhookHandler(channel_secret=CHANNEL_SECRET)

# การเชื่อมต่อ และตั้งค่าข้อมูลเพื่อเรียกใช้งาน Gemini
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")


# Endpoint สำหรับการสร้าง Webhook
@app.post('/message')
async def message(request: Request):
    # การตรวจสอบ headers จากการขอเรียกใช้บริการว่ามาจากทาง LINE Platform จริง
    signature = request.headers.get('X-Line-Signature')
    if not signature:
        raise HTTPException(
            status_code=400, detail="X-Line-Signature header is missing")

    # ข้อมูลที่ส่งมาจาก LINE Platform
    body = await request.body()

    try:
        # เรียกใช้งาน Handler เพื่อจัดข้อความจาก LINE Platform
        handler.handle(body.decode("UTF-8"), signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="Invalid signature")


# Function สำหรับจัดการข้อมูลที่ส่งมากจาก LINE Platform
@handler.add(MessageEvent, message=(TextMessageContent, ImageMessageContent))
def handle_message(event: MessageEvent):
    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)

        # ตรวจสอบ Message ว่าเป็นประเภทข้อความ Text
        if isinstance(event.message, TextMessageContent):
            # นำข้อมูลส่งไปยัง Gemini เพื่อทำการประมวลผล และสร้างคำตอบ และส่งตอบกลับมา
            gemini_response = model.generate_content(event.message.text)

            # Reply ข้อมูลกลับไปยัง LINE
            line_bot_api.reply_message_with_http_info(
                ReplyMessageRequest(
                    replyToken=event.reply_token,
                    messages=[TextMessage(text=gemini_response.text)]
                )
            )

        # ตรวจสอบ Message ว่าเป็นประเภทข้อความ Image
        if isinstance(event.message, ImageMessageContent):
            # การขอข้อมูลภาพจาก LINE Service
            headers = {"Authorization": f"Bearer {ACCESS_TOKEN}"}
            url = f"https://api-data.line.me/v2/bot/message/{event.message.id}/content"
            try:
                response = requests.get(url, headers=headers, stream=True)
                response.raise_for_status()
                image_data = BytesIO(response.content)
                image = Image.open(image_data)
            except Exception as e:
                line_bot_api.reply_message_with_http_info(
                    ReplyMessageRequest(
                        replyToken=event.reply_token,
                        messages=[TextMessage(text="เกิดข้อผิดพลาด, กรุณาลองใหม่อีกครั้ง🙏🏻")]
                    )
                )
                return

            try:
                gemini_response = model.generate_content(['อธิบายรูปภาพนี้', image])
                response_text = gemini_response.text
            except Exception as e:
                response_text = f"เกิดข้อผิดพลาด, ไม่สามารถประมวลผลรูปภาพได้"

            # Reply ข้อมูลกลับไปยัง LINE
            line_bot_api.reply_message_with_http_info(
                ReplyMessageRequest(
                    replyToken=event.reply_token, messages=[TextMessage(text=response_text)]
                )
            )

if __name__ == "__main__":
    uvicorn.run("main:app",
                port=8000,
                host="0.0.0.0")
