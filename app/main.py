from fastapi import FastAPI, status
from fastapi import File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.params import Depends
from sqlalchemy.orm import Session
from . import schema, model
from .database import engine, get_db
import cloudinary
import cloudinary.uploader

# Create the table in the database
model.Base.metadata.create_all(bind=engine)


app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/")
def read_root():
    return {"message": "Welcome to the face matching API"}

# Define endpoints for face matching
# Add routes for passport capture submission and matching



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
    return {"message": "Passport image submitted successfully"}


@app.post("/match-faces/")
async def match_faces(passport_image: UploadFile = File(...)):
    # Compare the submitted passport image with pre-existing images
    # Perform face matching and return matching score
    # Increment successful match count
    matching_score = 0.75  # Placeholder value
    return JSONResponse(content={"matching_score": matching_score})



