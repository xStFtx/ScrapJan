import requests
from bs4 import BeautifulSoup
import json
import csv
from transformers import pipeline
from typing import List, Dict, Union, Optional
import logging
from pathlib import Path
import numpy as np
from datetime import datetime
import re

# Configure logging with more detail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scraper.log'),
        logging.StreamHandler()
    ]
)

class WebScraper:
    def __init__(self):
        try:
            self.semantic_search = pipeline("feature-extraction", 
                                         model="sentence-transformers/all-MiniLM-L6-v2")
            self.sentiment_analyzer = pipeline("sentiment-analysis")
            self.summarizer = pipeline("summarization", max_length=100)
            logging.info("NLP models initialized successfully")
        except Exception as e:
            logging.error(f"Error initializing NLP models: {e}")
            raise

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text to improve matching quality.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Preprocessed text
        """
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^\w\s.,!?-]', ' ', text)
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        # Convert to lowercase
        text = text.lower().strip()
        return text

    def get_embedding(self, text: str, debug: bool = False) -> np.ndarray:
        """Get embedding for text by taking mean of token embeddings."""
        try:
            # Preprocess text
            processed_text = self.preprocess_text(text)
            if debug:
                logging.info(f"Processing text: {processed_text[:100]}...")
            
            # Get embeddings
            embeddings = self.semantic_search(processed_text)
            if isinstance(embeddings, list):
                embeddings = embeddings[0]
            
            # Take mean across tokens
            mean_embedding = np.mean(embeddings, axis=0)
            
            if debug:
                logging.info(f"Embedding shape: {mean_embedding.shape}")
            
            return mean_embedding
        except Exception as e:
            logging.error(f"Error getting embedding for text: {e}")
            return None

    def fetch_webpage(self, url: str) -> Optional[str]:
        """Fetch webpage content with proper headers and error handling."""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                         '(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            return response.text
        except Exception as e:
            logging.error(f"Error fetching URL {url}: {e}")
            return None

    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        if vec1 is None or vec2 is None:
            return 0.0
        try:
            similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            return float(similarity)  # Convert to native Python float
        except Exception as e:
            logging.error(f"Error calculating cosine similarity: {e}")
            return 0.0

    def analyze_text(self, text: str) -> Dict[str, Union[str, float]]:
        """Perform NLP analysis on text including sentiment and summarization."""
        try:
            sentiment = self.sentiment_analyzer(text)[0]
            
            if len(text.split()) > 30:
                try:
                    summary = self.summarizer(text)[0]["summary_text"]
                except Exception as e:
                    logging.warning(f"Summarization failed: {e}")
                    summary = text[:200] + "..."
            else:
                summary = text
            
            return {
                "sentiment_label": sentiment["label"],
                "sentiment_score": sentiment["score"],
                "summary": summary
            }
        except Exception as e:
            logging.error(f"Error in text analysis: {e}")
            return {
                "sentiment_label": "UNKNOWN",
                "sentiment_score": 0.0,
                "summary": text[:200] + "..."
            }

    def scrape_website(
        self, 
        url: str, 
        keywords: List[str], 
        elements_to_search: List[str] = ['p', 'h1', 'h2', 'h3', 'h4', 'span'],
        similarity_threshold: float = 0.2  # Lowered threshold
    ) -> List[Dict]:
        """Main scraping function that combines all functionality."""
        content = self.fetch_webpage(url)
        if not content:
            return []

        soup = BeautifulSoup(content, 'html.parser')
        
        # Get embeddings for keywords with debugging
        logging.info(f"Processing keywords: {keywords}")
        keyword_embeddings = []
        for keyword in keywords:
            embedding = self.get_embedding(keyword, debug=True)
            if embedding is not None:
                keyword_embeddings.append(embedding)
                logging.info(f"Successfully generated embedding for keyword: {keyword}")
            else:
                logging.warning(f"Failed to generate embedding for keyword: {keyword}")

        results = []
        processed_texts = set()  # To avoid duplicate content

        for element in elements_to_search:
            instances = soup.find_all(element)
            logging.info(f"Found {len(instances)} {element} elements")
            
            for instance in instances:
                text = instance.get_text(strip=True)
                if not text or len(text) < 10:
                    continue
                
                # Skip if we've already processed this exact text
                processed_text = self.preprocess_text(text)
                if processed_text in processed_texts:
                    continue
                processed_texts.add(processed_text)

                try:
                    text_embedding = self.get_embedding(text)
                    if text_embedding is None:
                        continue

                    similarity_scores = [
                        self.cosine_similarity(keyword_embedding, text_embedding)
                        for keyword_embedding in keyword_embeddings
                    ]

                    # Debug similarity scores
                    max_similarity = max(similarity_scores) if similarity_scores else 0
                    if max_similarity > 0.1:  # Log any reasonable matches for debugging
                        logging.info(f"Found potential match (score: {max_similarity:.3f}): {text[:100]}...")

                    if similarity_scores and max_similarity > similarity_threshold:
                        analysis = self.analyze_text(text)
                        results.append({
                            "element": element,
                            "text": text,
                            "sentiment": analysis["sentiment_label"],
                            "sentiment_score": analysis["sentiment_score"],
                            "summary": analysis["summary"],
                            "max_similarity": max_similarity,
                            "matched_keyword": keywords[np.argmax(similarity_scores)]
                        })
                except Exception as e:
                    logging.error(f"Error processing element {element}: {e}")
                    continue

        logging.info(f"Found {len(results)} matches above threshold {similarity_threshold}")
        return results

    def save_results(self, results: List[Dict], format: str = 'json') -> None:
        """Save results to file in specified format."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format.lower() == 'csv':
            filename = f'scrape_results_{timestamp}.csv'
            try:
                with open(filename, 'w', newline='', encoding='utf-8') as file:
                    writer = csv.writer(file)
                    writer.writerow(['Element', 'Text', 'Sentiment', 'Sentiment Score', 
                                   'Summary', 'Similarity Score', 'Matched Keyword'])
                    for result in results:
                        writer.writerow([
                            result["element"], result["text"], result["sentiment"],
                            result["sentiment_score"], result["summary"],
                            result["max_similarity"], result["matched_keyword"]
                        ])
                logging.info(f"Results saved to {filename}")
            except Exception as e:
                logging.error(f"Error saving CSV: {e}")
        
        elif format.lower() == 'json':
            filename = f'scrape_results_{timestamp}.json'
            try:
                with open(filename, 'w', encoding='utf-8') as file:
                    json.dump(results, file, ensure_ascii=False, indent=4)
                logging.info(f"Results saved to {filename}")
            except Exception as e:
                logging.error(f"Error saving JSON: {e}")

def main():
    """Main function to run the scraper interactively."""
    scraper = WebScraper()
    
    url = input("Enter the URL to scrape: ").strip()
    keywords = [k.strip() for k in input("Enter keywords (comma-separated): ").split(',')]
    elements = input("Enter HTML elements to search (comma-separated, default is p,h1,h2,h3,h4,span): ")
    
    elements = [e.strip() for e in elements.split(',')] if elements.strip() else \
              ['p', 'h1', 'h2', 'h3', 'h4', 'span']

    # Add timing information
    start_time = datetime.now()
    logging.info(f"Starting scrape of {url} for keywords: {keywords}")
    
    results = scraper.scrape_website(url, keywords, elements)
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    logging.info(f"Scraping completed in {duration:.2f} seconds")
    
    if results:
        print(f"\nFound {len(results)} matching results:")
        for i, result in enumerate(results, 1):
            print(f"\nResult {i}:")
            print(f"Element: <{result['element']}>")
            print(f"Text: {result['text'][:200]}...")
            print(f"Sentiment: {result['sentiment']} ({result['sentiment_score']:.2f})")
            print(f"Summary: {result['summary']}")
            print(f"Similarity Score: {result['max_similarity']:.2f}")
            print(f"Matched Keyword: {result['matched_keyword']}")
        
        save_format = input("\nSave results? (csv/json/n): ").strip().lower()
        if save_format in ['csv', 'json']:
            scraper.save_results(results, save_format)
    else:
        print("No matches found for the provided keywords.")

if __name__ == "__main__":
    main()