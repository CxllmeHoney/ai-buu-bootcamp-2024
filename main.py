import requests
from io import BytesIO
from PIL import Image
import uvicorn
import json
import numpy as np
import os
import faiss

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
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
from sentence_transformers import SentenceTransformer
from typing import Dict
from contextlib import asynccontextmanager


app = FastAPI()

# ข้อมูล token และ channel secret สำหรับ LINE
ACCESS_TOKEN = os.getenv("LINE_ACCESS_TOKEN", "29DtEHVbIAioorYrV9cpGqpB/mR0eyJ/Sn3M9kHHX81xZ27dvyeA0YXlyDtTtIIBNJ5cFhiJAWpDUw8C63caD/LcTybuzeYMgHqkyhWrl0kZZrHWNiazi3/0zluFQRwZSzAozVG4Iyd8zFLkTVvoKQdB04t89/1O/w1cDnyilFU=")
CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "270eef1263d1e55cf8f8d4abf041e3e9")

# ข้อมูล Gemini api key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyCZruMeHjowdcGne-dpfCfxJ4lVk5pnrPQ")

# การเชื่อมต่อ และตั้งค่าข้อมูลเพื่อเรียกใช้งาน LINE Messaging API
configuration = Configuration(access_token=ACCESS_TOKEN)
handler = WebhookHandler(channel_secret=CHANNEL_SECRET)

class GeminiRAGSystem:
    def __init__(self, 
                 json_db_path: str, 
                 gemini_api_key: str, 
                 embedding_model: str = 'all-MiniLM-L6-v2'):
        # การเชื่อมต่อ และตั้งค่าข้อมูลเพื่อเรียกใช้งาน Gemini
        genai.configure(api_key=gemini_api_key)
        
        # ประกาศโมเดลที่ใช้งาน
        self.generation_model = genai.GenerativeModel('gemini-1.5-flash')
        
        # ข้อมูล JSON ที่ใช้เก็บข้อมูล
        self.json_db_path = json_db_path
        
        # โมเดลที่ใช้ในการสร้างเวกเตอร์ของข้อความ
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Load ฐานข้อมูลจากไฟล์ JSON
        self.load_database()
        
        # สร้าง FAISS index
        self.create_faiss_index()
    
    def load_database(self):
        """Load existing database or create new"""
        try:
            with open(self.json_db_path, 'r', encoding='utf-8') as f:
                self.database = json.load(f)
        except FileNotFoundError:
            self.database = {
                'documents': [],
                'embeddings': [],
                'metadata': []
            }
    
    def save_database(self):
        """Save database to JSON file"""
        with open(self.json_db_path, 'w', encoding='utf-8') as f:
            json.dump(self.database, f, indent=2, ensure_ascii=False)
     
    def add_document(self, text: str, metadata: dict = None):
        """Add document to database with embedding"""
        # ประมวลผลข้อความเพื่อหาเวกเตอร์ของข้อความ
        embedding = self.embedding_model.encode([text])[0]
        
        # เพิ่มข้อมูลลงในฐานข้อมูล
        self.database['documents'].append(text)
        self.database['embeddings'].append(embedding.tolist())
        self.database['metadata'].append(metadata or {})
        
        # Save ฐานข้อมูลลงในไฟล์ JSON
        self.save_database()
        self.create_faiss_index()
    
    def create_faiss_index(self):
        """Create FAISS index for similarity search"""
        if not self.database['embeddings']:
            return
        
        # แปลงข้อมูลเป็น numpy array
        embeddings = np.array(self.database['embeddings'], dtype='float32')
        dimension = embeddings.shape[1]

        # สร้าง FAISS index
        self.index = faiss.IndexFlatL2(dimension)
        
        # เพิ่มข้อมูลลงใน FAISS index
        self.index.add(embeddings)
    
    def retrieve_documents(self, query: str, top_k: int = 3):
        """Retrieve most relevant documents"""
        if not self.database['embeddings']:
            return []
        
        # แปลงข้อความเป็นเวกเตอร์
        query_embedding = self.embedding_model.encode([query])
        
        # ค้นหาเอกสารที่เกี่ยวข้องด้วย similarity search
        D, I = self.index.search(query_embedding, top_k)
        
        return [self.database['documents'][i] for i in I[0]]
    
    def generate_response(self, query: str):
        """Generate response using Gemini and retrieved documents"""
        # Retrieve ข้ิอมูลจากฐานข้อมูล
        retrieved_docs = self.retrieve_documents(query)
        
        # เตรียมข้อมูลเพื่อใช้ในการสร้างคำถาม
        context = "\n\n".join(retrieved_docs)
        
        # สร้าง Prompt เพื่อใช้ในการสร้างคำตอบ
        full_prompt = f"""You are an AI assistant. 
        Use the following context to answer the question precisely:

        Context:
        {context}

        Question: {query}
        
        Provide a detailed and informative response based on the context in Thai 
        but if the response is not about the context just ignore and answering in way nat."""
        
        # คำตอบจาก Gemini
        try:
            response = self.generation_model.generate_content(full_prompt)
            return response.text, full_prompt
        except Exception as e:
            return f"Error generating response: {str(e)}", str(e)
    
    def process_image_query(self, 
                            image_content: bytes, 
                            query: str,
                            use_rag: bool = True,
                            top_k_docs: int = 3) -> Dict:
        """
        Process image-based query with optional RAG enhancement
        
        Args:
            image_content (bytes): Content of the image
            query (str): Query about the image
            use_rag (bool): Whether to use RAG for context
            top_k_docs (int): Number of documents to retrieve
        
        Returns:
            Generated response about the image
        """
        # เปิดภาพจากข้อมูลที่ส่งมา
        image = Image.open(BytesIO(image_content))

        # สร้างคำอธิบายของภาพ
        initial_description = self.generation_model.generate_content(
            ["Provide a detailed, objective description of this image", image],
            generation_config={
                "max_output_tokens": 256,
                "temperature": 0.4,
                "top_p": 0.9,
                "top_k": 8
            }
        ).text
        
        # สำหรับการใช้งาน RAG 
        context = ""
        if use_rag:
            # นำคำอธิบายภาพไปใช้ในการค้นหาเอกสารที่เกี่ยวข้องใน JSON
            retrieved_docs = self.retrieve_documents(initial_description, top_k_docs)
            
            # นำข้อมูลที่ได้จากการค้นหามาใช้ในการสร้างบริบท
            context = "\n\n".join(retrieved_docs)
        
        # สร้าง Prompt สำหรับการสร้างคำตอบ
        enhanced_prompt = f"""Image Description:
        {initial_description}

        Context from Knowledge Base:
        {context}

        User Query: {query}

        Based on the image description and the contextual information from our knowledge base, 
        provide a comprehensive and insightful response to the query. 
        If the context does not directly relate to the image, focus on the image description 
        and your visual analysis in Thai."""
        
        # สร้างคำตอบจาก Gemini
        try:
            response = self.generation_model.generate_content(
                [enhanced_prompt, image],
                generation_config={
                    "max_output_tokens": 256,
                    "temperature": 0.4,
                    "top_p": 0.9,
                    "top_k": 8
                }
            )
            
            return {
                "final_response": response.text,
            }
        except Exception as e:
            return {
                "error": f"Error generating response: {str(e)}",
                "image_description": initial_description
            }
    
    def clear_database(self):
        """Clear database and save to JSON file"""
        self.database = {
            'documents': [],
            'embeddings': [],
            'metadata': []
        }
        self.save_database()

# สร้าง Object สำหรับใช้งาน Gemini
gemini = GeminiRAGSystem(
    json_db_path="gemini_rag_database.json", 
    gemini_api_key=GEMINI_API_KEY
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ข้อมูลตัวอย่างที่ใช้สำหรับ Gemini
    sample_documents = [
        "ภีมวัจน์ คือ นักศึกษามหาวิทยาลัยบูรพา ปี 3 สาขา AAI ชื่อเล่นว่า ฮันนี่",
        "Clicknext",
        "งาน BUU-Bootcamp-2024 จัดที่มหาวิทยาลัยบูรพา ในวันที่ 25 มกราคม ปีค.ศ.2024 โดยมีการจัดกิจกรรมต่าง ๆ ที่เกี่ยวข้องกับการพัฒนาซอฟต์แวร์ เวลา 9:00 น. - 16:00 น.",
        "มหาวิทยาลัยบูรพา สาขาวิชาAI ปีการศึกษา 2565 มีนักศึกษาจำนวน 100 คน มีอาจารย์ที่ปรึกษา 10 คน",
        "clickNext team Ai คือ พี่ชวนทำ DataOps = Automation + Monitoring + Testing.,พี่ปาร์ม Back-end ,พี่ออย BA ,พี่เจ๋ง Front-end , พี่ดาว BA "
    ]
    
    # เพิ่มข้อมูลตัวอย่างลงใน Gemini
    for doc in sample_documents:
        gemini.add_document(doc)
        
    yield

    # ลบข้อมูลที่ใช้ในการทดสอบออกจาก Gemini
    gemini.clear_database()

app = FastAPI(lifespan=lifespan)

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
            gemini_response, prompt = gemini.generate_response(event.message.text)

            # Reply ข้อมูลกลับไปยัง LINE
            line_bot_api.reply_message_with_http_info(
                ReplyMessageRequest(
                    replyToken=event.reply_token,
                    messages=[TextMessage(text=gemini_response)]
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
            
            if image.size[0] * image.size[1] > 1024 * 1024:
                line_bot_api.reply_message_with_http_info(
                    ReplyMessageRequest(
                        replyToken=event.reply_token,
                        messages=[TextMessage(text="ขอโทษครับ ภาพมีขนาดใหญ่เกินไป กรุณาลดขนาดภาพและลองใหม่อีกครั้ง")]
                    )
                )

            try:
                # ส่งข้อมูลภาพไปยัง Gemini เพื่อทำการประมวลผล
                gemini_response = gemini.process_image_query(response.content,
                                                             query="อธิบายภาพนี้ให้ละเอียด", 
                                                             use_rag=True)
                # นำข้อมูลที่ได้จาก Gemini มาใช้งาน
                response_text = gemini_response['final_response']
            except Exception as e:
                response_text = f"เกิดข้อผิดพลาด, ไม่สามารถประมวลผลรูปภาพได้"

            # Reply ข้อมูลกลับไปยัง LINE
            line_bot_api.reply_message_with_http_info(
                ReplyMessageRequest(
                    replyToken=event.reply_token, messages=[TextMessage(text=response_text)]
                )
            )

# Endpoint สำหรับทดสอบ Gemini ด้วยข้อความ
@app.get('/test-message')
async def test_message_gemini(text: str):
    """
    Debug message from Gemini
    """
    response, prompt = gemini.generate_response(text)

    return {
        "gemini_answer": response,
        "full_prompt": prompt
    }

# Endpoint สำหรับทดสอบ Gemini ด้วยรูปภาพ
@app.post('/image-query')
async def image_query(
    file: UploadFile = File(...), 
    query: str = Form("อธิบายภาพนี้ให้ละเอียด"),
    use_rag: bool = Form(True)
):
    if file.size > 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image size too large")

    # ิ อ่านข้อมูลภาพจากไฟล์ที่ส่งมา
    contents = await file.read()

    # ส่งข้อมูลภาพไปยัง Gemini เพื่อทำการประมวลผล
    image_response = gemini.process_image_query(
        image_content=contents,
        query=query,
        use_rag=use_rag
    )
    
    return image_response

if __name__ == "__main__":
    uvicorn.run("main:app",
                port=8000,
                host="0.0.0.0")
