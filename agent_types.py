from pydantic import BaseModel
from typing import List, Optional

# These can be used in the future if we try to do reverse linkedin searches for contact info.
class ContactInfo(BaseModel):
    email: Optional[str] = None
    phone: Optional[str] = None

class CompanyResult(BaseModel):
    company_name: str
    linkedin_company_summary: str
    linkedin_company_url: str
    linkedin_company_img: str
    active_employee_profile_urls: List[str]
    
class EmployeeResult(BaseModel):
    linkedin_employee_name: str
    linkedin_employee_summary: str
    linkedin_employee_url: str
    linkedin_employee_img: str
    is_employed_by_company: bool
    