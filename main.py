import os
from openai import OpenAI
from configs import topics, languages
from dotenv import load_dotenv
import multiprocessing
import re
from urllib.parse import unquote

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Change this as necessary
URLS_PER_CONFIG = 5
OUTPUT_DIR = "language_outputs"


os.makedirs(OUTPUT_DIR, exist_ok=True)



def extract_urls(raw_text):
    # Find all strings that look like URLs
    matches = re.findall(r'(https?://[^\s\'",\]]+)', raw_text)
    return matches

def fetch_urls(args):
    topic, language = args
    language_name, region = language

    try:
        response = client.responses.create(
            model="gpt-4o-mini",
            tools=[{"type": "web_search_preview"}],
            input=(
                f"Search for {URLS_PER_CONFIG} URLs on the topic '{topic}' in the language '{language_name}' from {region}. "
                f"Return the result as a JSON array of exactly {URLS_PER_CONFIG} strings. "
                f"Look into the url and make sure it is relevant to {topic}"
                f"Each string must be a URL to content written in {language_name} and relevant to {region}. "
                f"Do not include any explanation, only the array of URLs."
            )
        )


        return language, topic, response.output_text

    except Exception as e:
        print(f"Error fetching for topic '{topic}' and language '{language_name}': {e}")
        return language, topic, ""

def main():
    for topic in topics:
        with multiprocessing.Pool(processes=min(8, len(languages))) as pool:
            results = pool.map(fetch_urls, [(topic, lang) for lang in languages])

        for language, topic_name, raw_output in results:
            if not raw_output:
                continue

            urls = extract_urls(raw_output)
            if not urls:
                continue

            file_path = os.path.join(OUTPUT_DIR, f"{language[0]}_{language[1]}.txt")
            with open(file_path, "a", encoding="utf-8") as f:
                f.write(f"Topic: {topic_name}\n")
                for url in urls:
                    f.write(f"{unquote(url)}\n")
                f.write("\n")

if __name__ == "__main__":
    main()