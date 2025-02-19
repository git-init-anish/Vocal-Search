import requests
import faiss
import numpy as np
import concurrent.futures
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from sentence_transformers import SentenceTransformer


GOOGLE_API_KEY = "AIzaSyA5MTtoKvgvv-jMsYJZT5VnR61woaGNArk"
SEARCH_ENGINE_ID = "64931ce0073504dc8"


embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed_text(text):
    return embedding_model.encode(text, convert_to_numpy=True)

def google_custom_search(query, num_results=10):
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "q": query,
        "key": GOOGLE_API_KEY,
        "cx": SEARCH_ENGINE_ID,
        "num": num_results
    }
    response = requests.get(url, params=params).json()
    results = response.get("items", [])
    return [result["link"] for result in results]

def extract_text_with_selenium(url):
    try:
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--disable-gpu")
        options.add_argument("--window-size=1920x1080")
        prefs = {
            "profile.managed_default_content_settings.images": 2,
            "profile.managed_default_content_settings.stylesheets": 2,
            "profile.managed_default_content_settings.fonts": 2,
            "profile.managed_default_content_settings.cookies": 2,
            "profile.managed_default_content_settings.popups": 2,
            "profile.managed_default_content_settings.geolocation": 2,
            "profile.managed_default_content_settings.media_stream": 2,
        }
        options.add_experimental_option("prefs", prefs)
        options.page_load_strategy = "eager"
        chromedriver_path = ChromeDriverManager().install()
        service = Service(chromedriver_path)
        driver = webdriver.Chrome(service=service, options=options)
        driver.get(url)
        try:
            content = WebDriverWait(driver, 5).until(
                EC.presence_of_element_located((By.TAG_NAME, "article"))
            ).text
        except:
            content = WebDriverWait(driver, 5).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            ).text
        driver.quit()
        return content[:5000] 
    except Exception as e:
        print(f"Selenium Error on {url}: {e}")
        return ""

def extract_text_multithreaded(urls, max_workers=5):
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {executor.submit(extract_text_with_selenium, url): url for url in urls}

        for future in concurrent.futures.as_completed(future_to_url):
            url = future_to_url[future]
            try:
                text = future.result()
                if text and len(text) > 200:  
                    results[url] = text
            except Exception as e:
                print(f"Error extracting {url}: {e}")
    return list(results.values())  

def store_and_retrieve(query, docs):
    vector_dimension = 384  
    index = faiss.IndexFlatL2(vector_dimension)
    doc_vectors = np.array([embed_text(doc) for doc in docs], dtype=np.float32)
    index.add(doc_vectors)
    query_vector = np.array([embed_text(query)], dtype=np.float32)
    _, indices = index.search(query_vector, k=min(3, len(docs)))
    return [docs[i] for i in indices[0]]  

def fetch_and_process(query):
    print(f"Searching Google for: {query}\n")
    urls = google_custom_search(query)
    docs = extract_text_multithreaded(urls, max_workers=5)
    if not docs:
        print("No valid documents extracted.")
        return ""  
    best_docs = store_and_retrieve(query, docs)
    return best_docs

