from fastapi import FastAPI, status
from fastapi import File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.params import Depends
from sqlalchemy.orm import Session
from .hog_face_detection import compare_faces
from .helpers import generate_passport_descriptors, get_one_face_descriptor
from . import schema, model
from .database import engine, get_db, SessionLocal
import cloudinary
import cloudinary.uploader

# Create the table in the database
model.Base.metadata.create_all(bind=engine)


# Define a variable to track the count of successful matches
successful_match_count = None
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
    global successful_match_count
    try:
        db_passport_descriptors = await generate_passport_descriptors_from_db(db)
        successful_match_count = 0
        print({"message": "Passport Face Descriptors Loaded in memory Successfully"})
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
    except Exception as e:
        pass
    new_user = model.User(**user.dict())
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    await startup_event()
    # return {"message": "Passport image submitted successfully"}
    return new_user



@app.post("/match-passport")
async def match_faces(file: UploadFile = File(...)):
    global successful_match_count
    distances = {}
    input_desc = await get_one_face_descriptor(file)
    for id, desc in db_passport_descriptors.items():
        distances[id] = compare_faces(input_desc, desc)
    min_key, min_value = min(distances.items(), key=lambda item: item[1])
    if min_value < 0.3:
        successful_match_count+=1
    print(distances)
    return JSONResponse(content={"matching_score": (1 - min_value) * 100, "match_user_id": min_key, "bio_match_count": successful_match_count})
   

