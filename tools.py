from langchain.tools import BaseTool
from typing import ClassVar, List, Optional
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import AsyncChromiumLoader
from pydantic import BaseModel

######## INITIAL SEARCH TOOL ########
search_tool = TavilySearchResults(
    max_results=10,
    include_answer=True,
    include_raw_content=True,
    include_images=True,
    search_depth="advanced",
    include_domains = []
    # exclude_domains = ["linkedin.com"]
)

######## RESULT QUALITY CHECKER TOOL ########
class ContactInfo(BaseModel):
    email: Optional[str] = None
    phone: Optional[str] = None

class SearchResult(BaseModel):
    client_name: str
    linkedin_title: str
    linkedin_image: Optional[str] = None
    linkedin_profile_url: str
    client_contact: ContactInfo
    company_name: str
    linkedin_company_summary: str
    linkedin_company_url: str
                         

class QualityCheckerTool(BaseTool):
    name: ClassVar[str] = "quality_checker"
    description: ClassVar[str] = "Evaluates the quality of search results based on accuracy and completeness."
    args_schema: ClassVar = SearchResult

    def _run(self, results: List[dict], **kwargs) -> tuple[List[dict], bool]:
        valid_list = []
        check_result = True
        for item in results:
            try:
                result = SearchResult(**item)
                print(result.model_dump())
                if " " not in result.client_name.strip():
                    raise Exception
                if "https://" not in result.linkedin_profile_url.strip():
                    raise Exception 
                if "N/A" in result.company_name:
                    raise Exception
                if "https://" not in result.linkedin_company_url.strip():
                    raise Exception 
                valid_list.append(result)
            except Exception:
                continue
        
        if len(valid_list) < 20:
            check_result = False

        return (valid_list, check_result)
    
    
######## URL CHECKER TOOL ########
class UrlChecker(BaseTool):
    name: ClassVar[str] = "url_checker"
    description: ClassVar[str] = "Scrapes a web page of the provided URL to determine if the page is valid."
    args_schema: ClassVar = SearchResult

    def _run(self):
        raise NotImplementedError("This tool is asynchronous. Use `_arun` instead.")

    async def _arun(self, results: List[SearchResult], **kwargs) -> tuple[List[dict], bool]:
        valid_list = []
        check_result = True

        for item in results:
            try:
                # Load profile page
                profile_loader = await AsyncChromiumLoader([item.linkedin_profile_url])
                profile_docs = await profile_loader.aload()
                profile_page_content = profile_docs[0].page_content if profile_docs else ""

                # Validate LinkedIn profile URL
                if "This LinkedIn Page isn't available" not in profile_page_content:
                    # Load company page
                    company_loader = await AsyncChromiumLoader([item.linkedin_company_url])
                    company_docs = await company_loader.aload()
                    company_page_content = company_docs[0].page_content if company_docs else ""

                    # Validate company page URL
                    if "This LinkedIn Page isn't available" not in company_page_content:
                        valid_list.append(item)
            except Exception:
                continue  # Skip invalid URLs

        # Check if fewer than 10 valid results were found
        if len(valid_list) < 10:
            check_result = False

        return ([result.dict() for result in valid_list], check_result)


                        

