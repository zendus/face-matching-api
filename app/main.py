from fastapi import FastAPI, status, HTTPException
from fastapi import File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.params import Depends
from sqlalchemy.orm import Session
from .hog_cnn_face_detection import compare_faces
from .helpers import generate_passport_descriptors, get_one_face_descriptor
from . import schema, model
from .database import engine, get_db, SessionLocal
import cloudinary
import cloudinary.uploader
import logging

# Create a logger instance
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG) 

# Create the table in the database
model.Base.metadata.create_all(bind=engine)


# Define a variable to track the count of successful matches
successful_match_count = 0
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


@app.get("/")
def read_root():
    return {"message": "Welcome to the face matching API"}



@app.post("/submit-passport", status_code=status.HTTP_201_CREATED)
async def submit_passport(file: UploadFile = File(...), user: schema.UploadPassport = Depends(schema.UploadPassport.as_form), 
db: Session = Depends(get_db)):
    try:
        passport_image = cloudinary.uploader.upload(file.file)
        url = passport_image.get("url")
        user.passport_url = url
        new_user = model.User(**user.dict())
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        await startup_event()
        return new_user
    except Exception as e:
        logger.error(f"Error occurred during passport image upload: {e}")
        raise HTTPException(status_code=500, detail="Error occurred during passport image upload")



@app.post("/match-passport")
async def match_faces(file: UploadFile = File(...)):
    try:
        global successful_match_count
        distances = {}
        input_desc = await get_one_face_descriptor(file)
        for id, desc in db_passport_descriptors.items():
            distances[id] = compare_faces(input_desc, desc)
        min_key, min_value = min(distances.items(), key=lambda item: item[1])
        if min_value < 0.4:
            successful_match_count+=1
            return JSONResponse(content={"matching_score": (1 - min_value) * 100, "match_user_id": min_key, "successful_bio_match_count": successful_match_count})
        else:
            return JSONResponse(content={"matching_score": (1 - min_value) * 100, "message": "Match Score less than threshhold (60%)", "successful_bio_match_count": successful_match_count})
            
    except Exception as e:
        logger.error(f"Error occurred during face match: {e}")
        raise HTTPException(status_code=500, detail="Error occurred during face match")
