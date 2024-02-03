from pydantic import BaseModel
from decorators import as_form
from typing import Optional


@as_form
class UploadPassport(BaseModel):
    name: str
    passport_url: Optional[str]


class UserOut(BaseModel):
    id: int
    name: str
    password_url: str