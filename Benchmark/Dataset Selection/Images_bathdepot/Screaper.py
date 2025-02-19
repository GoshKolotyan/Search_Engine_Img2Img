import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import time

URLs = {
    "Tub_URLs": [
        "https://www.bathdepot.com/bathtubs/all-bathtubs.html?page=1",
        "https://www.bathdepot.com/bathtubs/all-bathtubs.html?page=2",
        "https://www.bathdepot.com/bathtubs/all-bathtubs.html?page=3",
        "https://www.bathdepot.com/bathtubs/all-bathtubs.html?page=4",
    ],
    "Toilet_URLs": [
        "https://www.bathdepot.com/toilets/toilets.html?page=1",
        "https://www.bathdepot.com/toilets/toilets.html?page=2",
        "https://www.bathdepot.com/toilets/toilets.html?page=3",
    ],
    "Mirror_URLs": [
        "https://www.bathdepot.com/mirrors/mirrors.html?page=1",
        "https://www.bathdepot.com/mirrors/mirrors.html?page=2",
        "https://www.bathdepot.com/mirrors/mirrors.html?page=3",
        "https://www.bathdepot.com/mirrors/mirrors.html?page=4",
    ],
    "Cabinet_URLs": [
        "https://www.bathdepot.com/furniture/cabinets.html?page=1",
        "https://www.bathdepot.com/furniture/cabinets.html?page=2",
        "https://www.bathdepot.com/furniture/cabinets.html?page=3",
        "https://www.bathdepot.com/furniture/vanities.html?page=1",
        "https://www.bathdepot.com/furniture/vanities.html?page=2",
        "https://www.bathdepot.com/furniture/vanities.html?page=3",
        "https://www.bathdepot.com/furniture/vanities.html?page=4",
        "https://www.bathdepot.com/furniture/vanities.html?page=5",
    ],
    "Sink_URLs": [
        "https://www.bathdepot.com/sinks.html?page=1",
        "https://www.bathdepot.com/sinks.html?page=2",
        "https://www.bathdepot.com/sinks.html?page=3",
        "https://www.bathdepot.com/sinks.html?page=4",
        "https://www.bathdepot.com/sinks.html?page=5",
    ],
}

# Folder to save images
SAVE_FOLDER = "Images"
os.makedirs(SAVE_FOLDER, exist_ok=True)

# Setup Selenium WebDriver
options = webdriver.ChromeOptions()
options.add_argument("--headless")  # Run in headless mode
driver = webdriver.Chrome(
    service=Service(ChromeDriverManager().install()), options=options
)

# Iterate over each category and its associated URL list
for category, url_list in URLs.items():
    print(f"Processing {category}")

    # Create subfolder for this category
    category_folder = os.path.join(SAVE_FOLDER, category)
    os.makedirs(category_folder, exist_ok=True)

    # Go through each URL in the current category
    for URL in url_list:
        print("Loading URL:", URL)
        driver.get(URL)
        time.sleep(5)  # Allow JavaScript content to load

        # Scroll down to load lazy-loaded images
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(5)  # Wait for images to load

        soup = BeautifulSoup(driver.page_source, "html.parser")

        # Find product images using the correct class
        product_images = soup.select("img.item-ext-image-hLa")

        print(f"Found {len(product_images)} product images in {category}.")

        # Extract and save images
        for index, img in enumerate(product_images):
            img_url = (
                img.get("src")
                or img.get("data-src")
                or img.get("data-lazy")
                or img.get("srcset")
            )
            if img_url and not img_url.startswith("data:image/svg+xml"):  # Skip SVG images
                img_url = urljoin(URL, img_url)  # Ensure full URL
                print(f"Downloading: {img_url}")
                try:
                    img_data = requests.get(img_url).content
                    # Construct image filename
                    img_name = f"image_{index}_page_{URL.split('=')[-1]}.jpg"
                    # Save to the category subfolder
                    img_path = os.path.join(category_folder, img_name)
                    with open(img_path, "wb") as f:
                        f.write(img_data)
                    print(f"Saved: {img_name} in {category}")
                except Exception as e:
                    print(f"Failed to save {img_url}: {e}")
            else:
                print(f"Skipped: {img_url}")

driver.quit()
print("Scraping completed.")
