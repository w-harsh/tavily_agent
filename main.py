import getpass
import os
import requests
import json
from typing import Any, Dict, Optional, List
import datetime
from langchain_tavily import TavilySearch, TavilyExtract
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

HUGGINGFACE_API_KEY = ""
os.environ["TAVILY_API_KEY"] = ""


search_memory = ConversationBufferMemory()
extract_memory = ConversationBufferMemory()


class ResearchAgent:
    def __init__(self):
        self.search_tool = TavilySearch(
            max_results=5,  
            topic="general"
        )
        self.extract_tool = TavilyExtract(
            extract_depth="advanced", 
            include_images=False
        )
        self.memory = ConversationBufferMemory()
        
    def gather_information(self, query: str, urls: List[str] = None) -> Dict:
        """Gather information from multiple sources"""
        results = {
            "search_results": [],
            "extracted_info": [],
            "sources": []
        }
        
        search_results = self.search_tool.invoke(query)
        results["search_results"].append(search_results)
        
        if urls:
            for url in urls:
                extracted = self.extract_tool.invoke({"urls": [url]})
                results["extracted_info"].append(extracted)
                results["sources"].append(url)
        
        return results


class AnswerDraftingAgent:
    def __init__(self):
        self.memory = ConversationBufferMemory()
        
    def draft_answer(self, research_data: Dict, query: str) -> str:
        """Synthesize information into a coherent answer"""
       
        combined_info = f"Query: {query}\n\n"
        combined_info += "Research Findings:\n"
        
        for result in research_data["search_results"]:
            combined_info += f"- {str(result)}\n"
            
        for info in research_data["extracted_info"]:
            combined_info += f"- {str(info)}\n"
            
        combined_info += "\nSources:\n"
        for source in research_data["sources"]:
            combined_info += f"- {source}\n"
            
        
        payload = {
            "inputs": combined_info,
            "parameters": {
                "max_length": 500,  
                "min_length": 100,
                "do_sample": False
            }
        }
        
        response = query_huggingface(payload)
        return response[0]['summary_text'] if isinstance(response, list) else response.get('summary_text', str(response))

def query_huggingface(payload):
    API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
    response = requests.post(API_URL, headers=headers, json=payload)
    result = response.json()
    
    if isinstance(result, dict) and 'error' in result:
        print(f"\nWarning: API returned an error: {result.get('error')}")
        if 'warnings' in result:
            for warning in result.get('warnings', []):
                print(f"Warning: {warning}")
        return "Sorry, there was an error processing the content. Please try again with a different input."
    
    return result

def search(user_input):
    """Enhanced search function with multi-agent capabilities"""
    print("\nResearch Agent: Gathering information...")
    
   
    search_tool = TavilySearch(
        max_results=5, 
        topic="general"
    )
    search_results = search_tool.invoke(user_input)
    
   
    search_memory.save_context({"input": user_input}, {"output": str(search_results)})
    
    
    combined_info = f"Query: {user_input}\n\nResearch Findings:\n{str(search_results)}"
    
   
    if search_memory.chat_memory.messages:
        combined_info += "\n\nPrevious Context:\n" + str(search_memory.chat_memory.messages[-2:])
    
    payload = {
        "inputs": combined_info,
        "parameters": {
            "max_length": 500,  
            "min_length": 100,
            "do_sample": False
        }
    }
    
    print("\nAnswer Drafting Agent: Synthesizing information...")
    response = query_huggingface(payload)
    return response[0]['summary_text'] if isinstance(response, list) else response.get('summary_text', str(response))

def extract(user_input):
    """Enhanced extract function with multi-agent capabilities"""
    print("\nResearch Agent: Extracting information...")
    
    
    extract_tool = TavilyExtract(
        extract_depth="advanced",  
        include_images=False
    )

  
    is_url = user_input.startswith(('http://', 'https://'))
    if is_url:
        print(f"\nExtracting information from URL: {user_input}")
    else:
        print("\nExtracting information from text input")
  
    input_list = [user_input]
    search_results = extract_tool.invoke({"urls": input_list})
    
    extract_memory.save_context({"input": user_input}, {"output": str(search_results)})
    
    combined_info = f"Query: {user_input}\n\nExtracted Information:\n{str(search_results)}"
    

    if extract_memory.chat_memory.messages:
        combined_info += "\n\nPrevious Context:\n" + str(extract_memory.chat_memory.messages[-2:])
    
    if len(combined_info) > 1000:
        combined_info = combined_info[:1000] + "..."
    
    payload = {
        "inputs": combined_info,
        "parameters": {
            "max_length": 500,  
            "min_length": 100,
            "do_sample": False
        }
    }
    
    print("\nAnswer Drafting Agent: Synthesizing information...")
    response = query_huggingface(payload)
    return response[0]['summary_text'] if isinstance(response, list) else response.get('summary_text', str(response))

def display_menu():
    print("\n=== Research Tool Menu ===")
    print("1. Search (Comprehensive research with context)")
    print("2. Extract (Extract information from URL or text)")
    print("   - Enter a URL (starting with http:// or https://)")
    print("   - Or enter any text to extract information from")
    print("3. Exit")
    return input("Select an option (1-3): ")

def main():
    while True:
        choice = display_menu()
        
        if choice == "1":
            query = input("\nEnter your search query: ")
            print("\nInitiating research...")
            result = search(query)
            print("\nResearch Results:")
            print(result)
            
        elif choice == "2":
            print("\nYou can enter either:")
            print("- A URL (starting with http:// or https://)")
            print("- Or any text to extract information from")
            query = input("\nEnter URL or text: ")
            print("\nExtracting information...")
            result = extract(query)
            print("\nExtracted Information:")
            print(result)
            
        elif choice == "3":
            print("\nThank you for using the Research Tool!")
            break
            
        else:
            print("\nInvalid option. Please try again.")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()

