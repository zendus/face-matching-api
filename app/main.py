from fastapi import FastAPI, status, HTTPException
from fastapi import File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.params import Depends
from sqlalchemy.orm import Session
from .hog_cnn_face_detection import compare_faces
from .helpers import generate_passport_descriptors, generate_face_descriptor_from_file, generate_face_descriptor_from_url
from . import model
from .database import engine, get_db, SessionLocal
import cloudinary
import cloudinary.uploader
import logging

# Create a logger instance
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG) 

# Create the table in the database
model.Base.metadata.create_all(bind=engine)


# Define a variable to store passport descriptors in memory
db_passport_descriptors = None

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


async def generate_passport_descriptors_from_db(db: Session):
    users =  db.query(model.User).all()
    passport_descriptors =  await generate_passport_descriptors(users)
    return passport_descriptors


async def upload_image_to_cloudinary(file: UploadFile = File(...), db: Session = Depends(get_db)):
    passport_image = cloudinary.uploader.upload(file.file)
    url = passport_image.get("url")
    new_user = model.User(passport_url = url)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {"user": new_user}


# Define the event handler for app startup
@app.on_event("startup")
async def startup_event():
    db = SessionLocal()
    global db_passport_descriptors
    try:
        db_passport_descriptors = await generate_passport_descriptors_from_db(db)
        logger.warning("Passport Face Descriptors Loaded in memory Successfully")
    except Exception as e:
        logger.error(f"Error occurred during server startup: {e}")
    finally:
        db.close()


# Root function 
@app.get("/")
def read_root():
    return {"message": "Welcome to the face matching API"}


# Route to upload passport to cloudinary, generate descriptor and save to memory
@app.post("/submit-passport", status_code=status.HTTP_201_CREATED)
async def submit_passport(obj: dict = Depends(upload_image_to_cloudinary)):
    try:
        user = obj["user"]
        face_descriptor = await generate_face_descriptor_from_url(user.passport_url)
        db_passport_descriptors[user.id] = face_descriptor
        return user
    except Exception as e:
        logger.error(f"Error occurred during passport image upload: {e}")
        raise HTTPException(status_code=500, detail="Error occurred during passport image upload")


# Route to match new passport to existing passports 
@app.post("/match-passport")
async def match_faces(file: UploadFile = File(...), db: Session = Depends(get_db)):
    try:
        distances = {}
        match_count = db.query(model.BioMatchCount).first()
        input_desc = await generate_face_descriptor_from_file(file)
        for id, desc in db_passport_descriptors.items():
            distances[id] = compare_faces(input_desc, desc)
        min_key, min_value = min(distances.items(), key=lambda item: item[1])
        if min_value < 0.4:
            match_count.count += 1
            db.commit()
            db.refresh(match_count)
            return JSONResponse(content={"matching_score": (1 - min_value) * 100, "match_user_id": min_key, "successful_bio_match_count": match_count.count})
        else:
            return JSONResponse(content={"matching_score": (1 - min_value) * 100, "message": "Match Score less than threshhold (60%)", "successful_bio_match_count": match_count.count})       
    except Exception as e:
        logger.error(f"Error occurred during face match: {e}")
        raise HTTPException(status_code=500, detail="Error occurred during face match")
