import re
from pydantic import BaseModel, Field, field_validator

print("models.py is running")

class DateTimeModel(BaseModel): 
    date:str=Field(description="Properly formatted date", pattern=r'^\d{2}-\d{2}-\d{4} \d{2}:\d{2}$')
    
    @field_validator("date")
    def check_format_date(cls, v):
        if not re.match(r'^\d{2}-\d{2}-\d{4} \d{2}:\d{2}$', v):  # Ensures 'DD-MM-YYYY HH:MM' format
            raise ValueError("The date should be in format 'MM-DD-YYYY HH:MM'")
        return v
    
class DateModel(BaseModel):
    date: str = Field(description="Properly formatted date", pattern=r'^\d{2}-\d{2}-\d{4}$')
    @field_validator("date")
    def check_format_date(cls, v):
        if not re.match(r'^\d{2}-\d{2}-\d{4}$', v):  # Ensures MM-DD-YYYY format
            raise ValueError("The date must be in the format 'MM-DD-YYYY'")
        return v
     
class IdentificationNumberModel(BaseModel):
    id: int = Field(description="Identification number (10 long)")
    @field_validator("id")
    def check_format_id(cls, v):
        if not re.match(r'^\d{10}$', str(v)):  # Convert to string before matching
            raise ValueError("手機號碼請輸入10位數")
        return v