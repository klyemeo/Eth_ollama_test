import uvicorn
import base64
import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ultralytics import YOLO

# 1. โหลดโมเดล YOLO (โหลดครั้งแรกจะนานหน่อย)
model = YOLO("yolov8n.pt")  # 'n' คือรุ่นเล็กสุด (เร็ว) เปลี่ยนเป็น 'm' หรือ 'l' ถ้าอยากได้แม่นๆ

app = FastAPI()

# เปิดให้ HTML เข้าถึงได้ (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ImageRequest(BaseModel):
    image: str  # รับ Base64 string

@app.post("/detect")
async def detect_objects(req: ImageRequest):
    try:
        # 2. แปลง Base64 กลับเป็นรูปภาพ
        image_data = base64.b64decode(req.image)
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 3. ให้ YOLO ทำงาน
        results = model(img)
        
        # 4. ดึงชื่อวัตถุที่เจอออกมา
        detected_objects = []
        for r in results:
            for box in r.boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                confidence = float(box.conf[0])
                detected_objects.append(f"{class_name} ({confidence:.2f})")

        # สรุปผลส่งกลับไป
        if not detected_objects:
            return {"message": "ไม่พบวัตถุที่รู้จักในภาพนี้"}
        
        # นับจำนวนของที่เจอ
        summary = ", ".join(detected_objects)
        return {"message": f"YOLO เจอ: {summary}"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("🚀 YOLO Server running on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)