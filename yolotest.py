import uvicorn
import base64
import cv2
import numpy as np
from collections import Counter
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ultralytics import YOLO

# โหลดโมเดล
model = YOLO("yolov8n.pt") 

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ImageRequest(BaseModel):
    image: str

@app.post("/detect")
async def process_image(req: ImageRequest):
    try:
        # 1. แปลง Base64 เป็นรูปภาพ
        image_data = base64.b64decode(req.image)
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return {"error": "อ่านรูปภาพไม่ได้"}

        # 2. ให้ YOLO ทำงาน
        results = model(img)
        result = results[0] # เอาผลลัพธ์รูปแรก

        # 3. นับจำนวนวัตถุ
        detected_classes = []
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            detected_classes.append(class_name)
        
        # สรุปยอด (เช่น {'person': 2, 'car': 1})
        counts = Counter(detected_classes)
        summary_text = ", ".join([f"{name}: {count}" for name, count in counts.items()])
        
        if not summary_text:
            summary_text = "ไม่พบวัตถุ"

        # 4. วาดกรอบตำแหน่งลงในภาพ (Plotting)
        # นี่คือคำสั่งวิเศษที่จะวาดกรอบและชื่อให้เองเลย
        annotated_frame = result.plot()

        # 5. แปลงรูปที่วาดแล้วกลับเป็น Base64 เพื่อส่งคืน
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        annotated_base64 = base64.b64encode(buffer).decode('utf-8')

        return {
            "message": summary_text,      # ข้อความสรุปจำนวน
            "processed_image": annotated_base64, # รูปที่มีกรอบสี่เหลี่ยม
            "raw_counts": dict(counts)    # ข้อมูลดิบเผื่อเอาไปใช้ต่อ
        }

    except Exception as e:
        print(f"Error: {e}")
        return {"message": f"Error: {str(e)}"}

if __name__ == "__main__":
    print("🚀 YOLO Server + Counting Running...")
    uvicorn.run(app, host="0.0.0.0", port=8000)